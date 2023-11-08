import os
import gc
import sys
import PIL
import time
import torch
import scipy
import random
import pickle
import kornia
import argparse
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.distributed as dist

from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
# from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models, datasets
from torch.utils.data.distributed import DistributedSampler

from skimage.exposure import match_histograms

EPS = 1e-10
CEPS = torch.complex(torch.tensor(EPS),torch.tensor(EPS)).exp()

import sys
sys.path.append('/root/Cardiac-MRI-Reconstrucion/')

from utils.functions import fetch_loss_function

def complex_log(ct):
    indices = ct.abs() < 1e-10
    ct[indices] = CEPS
    return ct.log()

def myimshow(x, cmap = 'gray', trim = False):
    if trim:
        percentile_95 = np.percentile(x, 95)
        percentile_5 = np.percentile(x, 5)
        x[x > percentile_95] = percentile_95
        x[x < percentile_5] = percentile_5
    x = x - x.min()
    x = x/ (x.max() + EPS)
    plt.axis('off')
    plt.imshow(x, cmap = cmap)

def special_trim(x):
    percentile_95 = np.percentile(x.detach().cpu(), 95)
    percentile_5 = np.percentile(x.detach().cpu(), 5)
    x = x.clip(percentile_5, percentile_95)
    x = x - x.min().detach()
    x = x/ (x.max().detach() + EPS)
    return x

def show_difference_image(im1, im2):
    im1 = (im1 - im1.min())
    im1 = (im1 / (im1.max() + EPS))
    im2 = (im2 - im2.min())
    im2 = (im2 / (im2.max() + EPS))
    diff = (im1-im2)
    plt.axis('off')
    plt.imshow(np.abs(diff), cmap = 'plasma', vmin=0, vmax=0.25)
    # plt.colorbar()
    return np.abs(diff).reshape(-1)

class Trainer(nn.Module):
    def __init__(self, model, trainset, testset, parameters, device, ddp_rank, ddp_world_size, args):
        super(Trainer, self).__init__()
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.parameters = parameters
        self.device = device
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.args = args
        # self.scaler = GradScaler(enabled=self.parameters['Automatic_Mixed_Precision'])
        self.train_sampler = DistributedSampler(
                                self.trainset, 
                                num_replicas=self.ddp_world_size, 
                                rank=self.ddp_rank, 
                                shuffle=True, 
                                drop_last=False
                            )
        self.train_test_sampler = DistributedSampler(
                                self.trainset, 
                                num_replicas=self.ddp_world_size, 
                                rank=self.ddp_rank, 
                                shuffle=False, 
                                drop_last=False
                            )
        self.test_sampler = DistributedSampler(
                                self.testset, 
                                num_replicas=self.ddp_world_size, 
                                rank=self.ddp_rank, 
                                shuffle=False, 
                                drop_last=False
                            )
        self.trainloader = torch.utils.data.DataLoader(
                            self.trainset,
                            batch_size=self.parameters['train_batch_size'], 
                            shuffle = False,
                            num_workers = self.parameters['dataloader_num_workers'],
                            pin_memory = False,
                            drop_last = False,
                            sampler = self.train_sampler
                        )
        self.traintestloader = torch.utils.data.DataLoader(
                            self.trainset,
                            batch_size=self.parameters['test_batch_size'], 
                            shuffle = False,
                            num_workers = self.parameters['dataloader_num_workers'],
                            pin_memory = False,
                            drop_last = False,
                            sampler = self.train_test_sampler
                        )
        self.testloader = torch.utils.data.DataLoader(
                            self.testset,
                            batch_size=self.parameters['test_batch_size'], 
                            shuffle = False,
                            num_workers = self.parameters['dataloader_num_workers'],
                            pin_memory = False,
                            drop_last = False,
                            sampler = self.test_sampler
                        )

        if self.parameters['optimizer'] == 'Adam':
            self.optim = optim.Adam(self.model.parameters(), lr=self.parameters['lr'], betas=self.parameters['optimizer_params'])
            self.parameters['scheduler_params']['cycle_momentum'] = False
            
        if self.parameters['scheduler'] == 'None':
            self.scheduler = None
        if self.parameters['scheduler'] == 'StepLR':
            mydic = self.parameters['scheduler_params']
            if self.ddp_rank == 0:
                self.scheduler = optim.lr_scheduler.StepLR(self.optim, mydic['step_size'], gamma=mydic['gamma'], verbose=mydic['verbose'])
            else:
                self.scheduler = optim.lr_scheduler.StepLR(self.optim, mydic['step_size'], gamma=mydic['gamma'], verbose=False)
        if self.parameters['scheduler'] == 'CyclicLR':
            mydic = self.parameters['scheduler_params']
            if self.ddp_rank == 0:
                self.scheduler = optim.lr_scheduler.CyclicLR(self.optim, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'],  verbose=mydic['verbose'])
            else:
                self.scheduler = optim.lr_scheduler.CyclicLR(self.optim, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'])

        self.l1loss = fetch_loss_function('L1',self.device, self.parameters['loss_params'])
        self.l2loss = fetch_loss_function('L2',self.device, self.parameters['loss_params'])
        self.SSIM = kornia.metrics.SSIM(11)
        # if self.criterion_FT is not None:
        #     self.criterion_FT = self.criterion_FT.to(self.device)
        # if self.criterion_reconFT is not None:
        #     self.criterion_reconFT = self.criterion_reconFT.to(self.device)

    def time_analysis(self):
        with torch.no_grad():
            total_times = []
            times = []
            tqdm_object = tqdm(enumerate(self.testloader), total = len(self.testloader))
            for i, (indices, masks, og_video, coilwise_input, coils_used, periods) in tqdm_object:
            # for i, (indices, undersampled_fts, masks, og_coiled_fts, og_coiled_vids, og_video, periods) in tqdm_object:
                undersampled_fts = (torch.fft.fftshift(torch.fft.fft2(coilwise_input), dim = (-2,-1)) + CEPS)
                batch, num_frames, chan, numr, numc = undersampled_fts.shape
                self.model.eval()
                predr = torch.zeros(batch, num_frames-6,numr,numc)
                current_fts = (undersampled_fts[:,:6]).to(self.device)
                for fii in range(num_frames - 6):
                    start = time.time()
                    current_fts = torch.cat((current_fts, (undersampled_fts[:,fii+6:fii+7]).to(self.device).log()), 1)
                    # current_fts = (undersampled_fts[:,fii:fii+7]).to(self.device).log()
                    current_ftss = torch.stack((current_fts.real,current_fts.imag), -1)
                    current_ftss = torch.swapaxes(current_ftss, 1,2)
                    _, temp = self.model(current_ftss)
                    predr[:,fii] = temp.squeeze().cpu()
                    current_fts = current_fts[:,1:]
                    times.append(time.time()-start)

        print('Average Time Per Frame = {} +- {}'.format(np.mean(times), np.std(times)), flush = True)
        scipy.io.savemat(os.path.join(self.args.run_id, 'fps.mat'), {'times': times})
        return


    def train(self, epoch, print_loss = False):
        avgispaceloss = 0.
        ispacessim_score = 0.
        avgispace_l1_loss = 0.
        avgispace_l2_loss = 0.
        self.trainloader.sampler.set_epoch(epoch)
        if self.ddp_rank == 0:
            tqdm_object = tqdm(enumerate(self.trainloader), total = len(self.trainloader), desc = "[{}] | KS Epoch {}".format(os.getpid(), epoch), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
        else:
            tqdm_object = enumerate(self.trainloader)
        for i, (indices, masks, og_video, coilwise_input, coils_used, periods) in tqdm_object:
        # for i, (indices, undersampled_fts, masks, og_coiled_fts, og_coiled_vids, og_video, periods) in tqdm_object:
            undersampled_fts = (torch.fft.fftshift(torch.fft.fft2(coilwise_input.to(self.device)), dim = (-2,-1)) + CEPS).log()
            undersampled_fts = torch.stack((undersampled_fts.real,undersampled_fts.imag), -1)
            undersampled_fts = torch.swapaxes(undersampled_fts, 1,2)
            # og_coiled_vids = og_video.to(self.device) * coils_used.to(self.device)
            # og_coiled_fts = torch.fft.fftshift(torch.fft.fft2(og_coiled_vids), dim = (-2,-1))

            
            batch, chan, num_frames, numr, numc, _ = undersampled_fts.shape
            self.model.train()
            totloss = 0
            predr = torch.zeros(batch, num_frames-6,numr,numc).to(self.device)
            for fii in range(num_frames - 6):
                self.optim.zero_grad(set_to_none=True)
                _, temp = self.model((undersampled_fts[:,:,fii:fii+7]))
                loss = self.l2loss(temp, (og_video[:,fii+6]).to(self.device))
                predr[:,fii] = temp.squeeze()
                loss.backward()
                totloss += loss.item()
                self.optim.step()
            del masks

            targ_vid = (og_video[:,3:num_frames-3]).squeeze().to(self.device)
            loss_l1 = (predr- targ_vid).reshape(predr.shape[0]*predr.shape[1], predr.shape[2]*predr.shape[3]).abs().mean(1).sum().detach().cpu()
            loss_l2 = (((predr- targ_vid).reshape(predr.shape[0]*predr.shape[1], predr.shape[2]*predr.shape[3]) ** 2).mean(1).sum()).detach().cpu()
            ss1 = self.SSIM(predr.reshape(predr.shape[0]*predr.shape[1],1,*predr.shape[2:]), targ_vid.reshape(predr.shape[0]*predr.shape[1],1,*predr.shape[2:]))
            ss1 = ss1.reshape(ss1.shape[0],-1)
            loss_ss1 = ss1.mean(1).sum().detach().cpu()

            avgispaceloss += float(totloss/(len(self.trainloader)))
            ispacessim_score += float(loss_ss1.cpu().item()/self.trainset.total_unskipped_frames)
            avgispace_l1_loss += float(loss_l1.cpu().item()/self.trainset.total_unskipped_frames)
            avgispace_l2_loss += float(loss_l2.cpu().item()/self.trainset.total_unskipped_frames)

        if self.scheduler is not None:
            self.scheduler.step()
        return avgispaceloss, ispacessim_score, avgispace_l1_loss, avgispace_l2_loss

    def evaluate(self, epoch, train = False, print_loss = False):
        avgispaceloss = 0.
        ispacessim_score = 0.
        avgispace_l1_loss = 0.
        avgispace_l2_loss = 0.
        if train:
            self.traintestloader.sampler.set_epoch(epoch)
            dloader = self.traintestloader
            dset = self.trainset
            dstr = 'Train'
        else:
            self.testloader.sampler.set_epoch(epoch)
            dloader = self.testloader
            dset = self.testset
            dstr = 'Test'
        if self.args.write_csv:
            f = open(os.path.join(self.args.run_id, '{}_results.csv'.format(dstr)), 'w')
            f.write('patient_number,')
            f.write('location_number,')
            f.write('frame_number,')
            f.write('SSIM,')
            f.write('L1,')
            f.write('L2')
            f.write('\n')
            f.flush()
        if self.ddp_rank == 0:
            tqdm_object = tqdm(enumerate(dloader), total = len(dloader), desc = "Testing after Epoch {} on {}set".format(epoch, dstr), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
        else:
            tqdm_object = enumerate(dloader)
        with torch.no_grad():
            for i, (indices, masks, og_video, coilwise_input, coils_used, periods) in tqdm_object:
            # for i, (indices, undersampled_fts, masks, og_coiled_fts, og_coiled_vids, og_video, periods) in tqdm_object:
                undersampled_fts = (torch.fft.fftshift(torch.fft.fft2(coilwise_input.to(self.device)), dim = (-2,-1)) + CEPS).log()
                undersampled_fts = torch.stack((undersampled_fts.real,undersampled_fts.imag), -1)
                undersampled_fts = torch.swapaxes(undersampled_fts, 1,2)
                # og_coiled_vids = og_video.to(self.device) * coils_used.to(self.device)
                # og_coiled_fts = torch.fft.fftshift(torch.fft.fft2(og_coiled_vids), dim = (-2,-1))

                batch, chan, num_frames, numr, numc, _ = undersampled_fts.shape
                self.model.eval()
                predr = torch.zeros(batch, num_frames-6,numr,numc).to(self.device)
                totloss = 0
                for fii in range(num_frames - 6):
                    predft, temp = self.model((undersampled_fts[:,:,fii:fii+7]))
                    loss = self.l2loss(temp, (og_video[:,fii+6]).to(self.device))
                    predr[:,fii] = temp.squeeze()
                    totloss += loss.item()
                targ_vid = (og_video[:,3:num_frames-3]).squeeze().to(self.device)

                if self.args.numbers_crop:
                    predr = predr[:,:,96:160]
                    targ_vid = targ_vid[:,:,96:160]

                if self.args.motion_mask:
                    red_num_frames = num_frames-6
                    motion_mask = torch.diff(targ_vid.reshape(batch, red_num_frames,-1)[:,np.arange(red_num_frames+1)%(red_num_frames)], n = 1, dim = 1).reshape(batch*red_num_frames,1,numr, numc).to(self.device)
                    motion_mask_min = motion_mask.min(1)[0].min(1)[0].min(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    motion_mask_max = motion_mask.max(1)[0].max(1)[0].max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    motion_mask = ((motion_mask - motion_mask_min)/(motion_mask_max+EPS)).reshape(targ_vid.shape)
                else:
                    motion_mask = torch.ones(targ_vid.shape).to(self.device)

                loss_l1 = ((predr*motion_mask)- (targ_vid*motion_mask)).reshape(predr.shape[0]*predr.shape[1], predr.shape[2]*predr.shape[3]).abs().mean(1).detach().cpu()
                loss_l2 = ((((predr*motion_mask)- (targ_vid*motion_mask)).reshape(predr.shape[0]*predr.shape[1], predr.shape[2]*predr.shape[3]) ** 2).mean(1)).detach().cpu()
                ss1 = self.SSIM((predr*motion_mask).reshape(predr.shape[0]*predr.shape[1],1,*predr.shape[2:]), (targ_vid*motion_mask).reshape(predr.shape[0]*predr.shape[1],1,*predr.shape[2:]))
                if self.args.write_csv:
                    for bi in range(batch):
                        for fi in range(num_frames-6):
                            f.write('{},'.format(indices[bi,0]))
                            f.write('{},'.format(indices[bi,1]))
                            f.write('{},'.format(fi))
                            f.write('{},'.format(ss1.reshape(batch, num_frames-6,-1).mean(2)[bi,fi]))
                            f.write('{},'.format(loss_l1.reshape(batch, num_frames-6)[bi,fi]))
                            f.write('{},'.format(loss_l2.reshape(batch, num_frames-6)[bi,fi]))
                            f.write('\n')
                            f.flush()

                ss1 = ss1.reshape(ss1.shape[0],-1)
                loss_ss1 = ss1.mean(1).sum().detach().cpu()
                loss_l1 = loss_l1.sum()
                loss_l2 = loss_l2.sum()

                avgispaceloss += float(totloss/(len(dloader)))
                ispacessim_score += float(loss_ss1.cpu().item()/(dset.total_unskipped_frames/8))
                avgispace_l1_loss += float(loss_l1.cpu().item()/(dset.total_unskipped_frames/8))
                avgispace_l2_loss += float(loss_l2.cpu().item()/(dset.total_unskipped_frames/8))

            if self.scheduler is not None:
                self.scheduler.step()

        if self.args.write_csv:
            f.flush()
            f.close()

        return avgispaceloss, ispacessim_score, avgispace_l1_loss, avgispace_l2_loss

    def visualise(self, epoch, train = False):

        if train:
            dloader = self.traintestloader
            dset = self.trainset
            dstr = 'Train'
            path = os.path.join(self.args.run_id, './images/train')
        else:
            dloader = self.testloader
            dset = self.testset
            dstr = 'Test'
            path = os.path.join(self.args.run_id, './images/test')
        
        print('Saving plots for {} data'.format(dstr), flush = True)
        with torch.no_grad():
            for i, (indices, masks, og_video, coilwise_input, coils_used, periods) in enumerate(dloader):
            # for i, (indices, undersampled_fts, masks, og_coiled_fts, og_coiled_vids, og_video, periods) in tqdm_object:
                if i == 2:
                    break
                undersampled_fts = (torch.fft.fftshift(torch.fft.fft2(coilwise_input.to(self.device)), dim = (-2,-1)) + CEPS).log().cpu()
                undersampled_fts = torch.stack((undersampled_fts.real,undersampled_fts.imag), -1)
                undersampled_fts = torch.swapaxes(undersampled_fts, 1,2)
                # og_coiled_vids = og_video.to(self.device) * coils_used.to(self.device)
                # og_coiled_fts = torch.fft.fftshift(torch.fft.fft2(og_coiled_vids), dim = (-2,-1))
                self.model.eval()

                batch, num_coils, num_frames, numr, numc, _ = undersampled_fts.shape
                
                tot_vids_per_patient = (dset.num_vids_per_patient*dset.frames_per_vid_per_patient)
                num_vids = 1
                batch = num_vids
                num_plots = num_vids*num_frames

                predr = torch.zeros(batch, num_frames-6,numr,numc)
                for fii in range(num_frames - 6):
                    predft, temp = self.model((undersampled_fts[:batch,:,fii:fii+7]).to(self.device))
                    predr[:,fii] = temp.squeeze().cpu()
                targ_vid = (og_video[:,3:num_frames-3]).squeeze().cpu()
                
                tot = 0
                with tqdm(total=num_plots) as pbar:
                    for bi in range(num_vids):
                        p_num, v_num = indices[bi]
                        for f_num in range(num_frames-6):

                            if not os.path.exists(os.path.join(path, './patient_{}/'.format(p_num))):
                                os.mkdir(os.path.join(path, './patient_{}/'.format(p_num)))
                            if not os.path.exists(os.path.join(path, './patient_{}/by_location_number/'.format(p_num))):
                                os.mkdir(os.path.join(path, './patient_{}/by_location_number/'.format(p_num)))
                            if not os.path.exists(os.path.join(path, './patient_{}/by_location_number/location_{}'.format(p_num, v_num))):
                                os.mkdir(os.path.join(path, './patient_{}/by_location_number/location_{}'.format(p_num, v_num)))
                            if not os.path.exists(os.path.join(path, './patient_{}/by_frame_number/'.format(p_num))):
                                os.mkdir(os.path.join(path, './patient_{}/by_frame_number/'.format(p_num)))
                            if not os.path.exists(os.path.join(path, './patient_{}/by_frame_number/frame_{}'.format(p_num, f_num))):
                                os.mkdir(os.path.join(path, './patient_{}/by_frame_number/frame_{}'.format(p_num, f_num)))
                            
                            fig = plt.figure(figsize = (12,6))
                            og_vidi = og_video.cpu()[bi, f_num,0,:,:]
                            ispace_outpi = predr[bi, f_num, :,:]
                            
                            plt.subplot(1,3,1)
                            myimshow(og_vidi, cmap = 'gray')
                            plt.title('Ground Truth Frame')
                            # print(f_num, 1)
                            # ispace_outpi = torch.from_numpy(match_histograms(ispace_outpi.numpy(), og_vidi.numpy(), channel_axis=None))
                            plt.subplot(1,3,2)
                            myimshow(ispace_outpi, cmap = 'gray')
                            plt.title('MDCNN Prediction')
                            # print(f_num, 2)
                            
                            plt.subplot(1,3,3)
                            diffvals = show_difference_image(ispace_outpi, og_vidi)
                            plt.title('Difference Frame')
                            # print(f_num, 3)

                            # plt.subplot(2,2,4)
                            # plt.hist(diffvals, range = [0,0.25], density = False, bins = 30, histtype='step')
                            # plt.ylabel('Pixel Count')
                            # plt.xlabel('Difference Value')
                            # print(f_num, 4)

                            spec = ''
                            plt.suptitle("Epoch {}\nPatient {} Video {} Frame {}\n{}".format(epoch,p_num, v_num, f_num+3, spec))
                            plt.savefig(os.path.join(path, './patient_{}/by_location_number/location_{}/io_frame_{}.jpg'.format(p_num, v_num, f_num)))
                            plt.savefig(os.path.join(path, './patient_{}/by_frame_number/frame_{}/io_location_{}.jpg'.format(p_num, f_num, v_num)))


                            plt.close('all')

                            tot += 1
                            pbar.update(1)
