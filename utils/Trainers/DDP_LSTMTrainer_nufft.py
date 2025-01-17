import os
import gc
import sys
import PIL
import time
import scipy
import torch
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
    def __init__(self, kspace_model, ispace_model, trainset, testset, parameters, device, ddp_rank, ddp_world_size, args):
        super(Trainer, self).__init__()
        self.kspace_model = kspace_model
        self.ispace_model = ispace_model
        self.trainset = trainset
        self.testset = testset
        self.parameters = parameters
        self.device = device
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.args = args
        temp = os.getcwd().split('/')
        temp = temp[temp.index('experiments'):]
        self.save_path = os.path.join(self.parameters['save_folder'], '/'.join(temp))
        self.save_path = os.path.join(self.save_path, self.args.run_id)
        del temp
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
            if self.parameters['kspace_architecture'] == 'KLSTM1':
                if self.parameters['ispace_lstm']:
                    self.kspace_optim_mag = optim.Adam(list(self.kspace_model.module.mag_m.parameters())+list(self.kspace_model.module.ispacem.parameters()), lr=self.parameters['lr_kspace_mag'], betas=self.parameters['optimizer_params'])
                else:
                    self.kspace_optim_mag = optim.Adam(self.kspace_model.module.mag_m.parameters(), lr=self.parameters['lr_kspace_mag'], betas=self.parameters['optimizer_params'])
                self.kspace_optim_phase = optim.Adam(self.kspace_model.module.phase_m.parameters(), lr=self.parameters['lr_kspace_phase'], betas=self.parameters['optimizer_params'])
            elif self.parameters['kspace_architecture'] == 'KLSTM2':
                self.kspace_optim = optim.Adam(self.kspace_model.module.parameters(), lr=self.parameters['lr_kspace_phase'], betas=self.parameters['optimizer_params'])
            self.ispace_optim = optim.Adam(self.ispace_model.parameters(), lr=self.parameters['lr_ispace'], betas=self.parameters['optimizer_params'])
            self.parameters['scheduler_params']['cycle_momentum'] = False
            
        if self.parameters['scheduler'] == 'None':
            self.kspace_scheduler = None
            self.ispace_scheduler = None
        if self.parameters['scheduler'] == 'StepLR':
            mydic = self.parameters['scheduler_params']
            if self.ddp_rank == 0:
                if self.parameters['kspace_architecture'] == 'KLSTM1':
                    self.kspace_scheduler_mag = optim.lr_scheduler.StepLR(self.kspace_optim_mag, mydic['step_size'], gamma=mydic['gamma'], verbose=mydic['verbose'])
                    self.kspace_scheduler_phase = optim.lr_scheduler.StepLR(self.kspace_optim_phase, mydic['step_size'], gamma=mydic['gamma'], verbose=mydic['verbose'])
                elif self.parameters['kspace_architecture'] == 'KLSTM2':
                    self.kspace_scheduler = optim.lr_scheduler.StepLR(self.kspace_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=mydic['verbose'])
                self.ispace_scheduler = optim.lr_scheduler.StepLR(self.ispace_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=mydic['verbose'])
            else:
                if self.parameters['kspace_architecture'] == 'KLSTM1':
                    self.kspace_scheduler_mag = optim.lr_scheduler.StepLR(self.kspace_optim_mag, mydic['step_size'], gamma=mydic['gamma'], verbose=False)
                    self.kspace_scheduler_phase = optim.lr_scheduler.StepLR(self.kspace_optim_phase, mydic['step_size'], gamma=mydic['gamma'], verbose=False)
                elif self.parameters['kspace_architecture'] == 'KLSTM2':
                    self.kspace_scheduler = optim.lr_scheduler.StepLR(self.kspace_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=False)
                self.ispace_scheduler = optim.lr_scheduler.StepLR(self.ispace_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=False)
        if self.parameters['scheduler'] == 'CyclicLR':
            mydic = self.parameters['scheduler_params']
            if self.ddp_rank == 0:
                if self.parameters['kspace_architecture'] == 'KLSTM1':
                    self.kspace_scheduler_mag = optim.lr_scheduler.CyclicLR(self.kspace_optim_mag, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'],  verbose=mydic['verbose'])
                    self.kspace_scheduler_phase = optim.lr_scheduler.CyclicLR(self.kspace_optim_phase, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'],  verbose=mydic['verbose'])
                elif self.parameters['kspace_architecture'] == 'KLSTM2':
                    self.kspace_scheduler = optim.lr_scheduler.CyclicLR(self.kspace_optim, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'],  verbose=mydic['verbose'])
                self.ispace_scheduler = optim.lr_scheduler.CyclicLR(self.ispace_optim, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'],  verbose=mydic['verbose'])
            else:
                if self.parameters['kspace_architecture'] == 'KLSTM1':
                    self.kspace_scheduler_mag = optim.lr_scheduler.CyclicLR(self.kspace_optim_mag, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'])
                    self.kspace_scheduler_phase = optim.lr_scheduler.CyclicLR(self.kspace_optim_phase, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'])
                elif self.parameters['kspace_architecture'] == 'KLSTM2':
                    self.kspace_scheduler = optim.lr_scheduler.CyclicLR(self.kspace_optim, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'])
                self.ispace_scheduler = optim.lr_scheduler.CyclicLR(self.ispace_optim, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'])

        self.l1loss = fetch_loss_function('L1',self.device, self.parameters['loss_params'])
        # self.l2loss = fetch_loss_function('L2',self.device, self.parameters['loss_params'])
        self.SSIM = kornia.metrics.SSIM(11)
        # if self.criterion_FT is not None:
        #     self.criterion_FT = self.criterion_FT.to(self.device)
        # if self.criterion_reconFT is not None:
        #     self.criterion_reconFT = self.criterion_reconFT.to(self.device)

    def train(self, epoch, print_loss = False):
        avgkspacelossphase = 0.
        avgkspacelossreal = 0.
        avgkspacelossmag = 0.
        kspacessim_score = 0.
        avgkspace_l1_loss = 0.
        avgkspace_l2_loss = 0.
        avgispacelossreal = 0.
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
            undersampled_fts = torch.fft.fftshift(torch.fft.fft2(coilwise_input.to(self.device)), dim = (-2,-1))
            og_coiled_vids = og_video.to(self.device) * coils_used.to(self.device)
            og_coiled_fts = torch.fft.fftshift(torch.fft.fft2(og_coiled_vids), dim = (-2,-1))

            if self.parameters['kspace_architecture'] == 'KLSTM1':
                self.kspace_optim_mag.zero_grad(set_to_none=True)
                self.kspace_optim_phase.zero_grad(set_to_none=True)
            elif self.parameters['kspace_architecture'] == 'KLSTM2':
                self.kspace_optim.zero_grad(set_to_none=True)

            
            batch, num_frames, chan, numr, numc = undersampled_fts.shape
            inpt_mag_log = (og_coiled_fts.abs()+EPS).log()
            inpt_phase = og_coiled_fts / inpt_mag_log.exp()
            inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)
            self.kspace_model.train()
            predr, _, _, loss_mag, loss_phase, loss_real, (loss_l1, loss_l2, ss1) = self.kspace_model(undersampled_fts, masks, self.device, periods.clone(), targ_phase = inpt_phase, targ_mag_log = inpt_mag_log, targ_real = og_coiled_vids, og_video = og_video)
            del masks
            del inpt_phase
            del inpt_mag_log
            if not self.parameters['end-to-end-supervision']:
                if self.parameters['kspace_real_loss_only']:
                    loss = 10*loss_real
                else:
                    loss = 0.06*loss_mag + loss_phase + 5*loss_real
                    # print(0.05*loss_mag,loss_phase,5*loss_real)

                loss.backward()
                if self.parameters['kspace_architecture'] == 'KLSTM1':
                    self.kspace_optim_mag.step()
                    self.kspace_optim_phase.step()
                elif self.parameters['kspace_architecture'] == 'KLSTM2':
                    self.kspace_optim.step()

                del loss
            

            avgkspacelossphase += float(loss_phase.cpu().item()/(len(self.trainloader)))
            avgkspacelossmag += float(loss_mag.cpu().item()/(len(self.trainloader)))
            avgkspacelossreal += float(loss_real.cpu().item()/(len(self.trainloader)))
            del loss_phase
            del loss_mag
            del loss_real
            
            kspacessim_score += float(ss1.cpu()/self.trainset.total_unskipped_frames)
            avgkspace_l1_loss += float(loss_l1.cpu()/self.trainset.total_unskipped_frames)
            avgkspace_l2_loss += float(loss_l2.cpu()/self.trainset.total_unskipped_frames)


            self.ispace_optim.zero_grad(set_to_none=True)
            if not self.parameters['end-to-end-supervision']:
                predr = predr.detach()[:,self.parameters['init_skip_frames']:]
            else:
                predr = predr[:,self.parameters['init_skip_frames']:]

            num_frames = num_frames - self.parameters['init_skip_frames']
            if self.parameters['kspace_coil_combination'] or self.parameters['ispace_lstm']:
                predr = predr.reshape(batch*num_frames,1,numr, numc).to(self.device)
            else:
                predr = predr.reshape(batch*num_frames,chan,numr, numc).to(self.device)
            targ_vid = og_video[:,self.parameters['init_skip_frames']:].reshape(batch*num_frames,1, numr, numc).to(self.device)

            outp = self.ispace_model(predr)
            loss = self.l1loss(outp, targ_vid)
            loss.backward()
            self.ispace_optim.step()

            if self.parameters['end-to-end-supervision']:
                if self.parameters['kspace_architecture'] == 'KLSTM1':
                    self.kspace_optim_mag.step()
                    self.kspace_optim_phase.step()
                elif self.parameters['kspace_architecture'] == 'KLSTM2':
                    self.kspace_optim.step()


            loss_l1 = (outp- targ_vid).reshape(outp.shape[0]*outp.shape[1], outp.shape[2]*outp.shape[3]).abs().mean(1).sum().detach().cpu()
            loss_l2 = (((outp- targ_vid).reshape(outp.shape[0]*outp.shape[1], outp.shape[2]*outp.shape[3]) ** 2).mean(1).sum()).detach().cpu()
            ss1 = self.SSIM(outp.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]), targ_vid.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]))
            ss1 = ss1.reshape(ss1.shape[0],-1)
            loss_ss1 = ss1.mean(1).sum().detach().cpu()

            avgispacelossreal += float(loss.cpu().item()/(len(self.trainloader)))
            ispacessim_score += float(loss_ss1.cpu().item()/self.trainset.total_unskipped_frames)
            avgispace_l1_loss += float(loss_l1.cpu().item()/self.trainset.total_unskipped_frames)
            avgispace_l2_loss += float(loss_l2.cpu().item()/self.trainset.total_unskipped_frames)

        if self.parameters['kspace_architecture'] == 'KLSTM1':
            if self.kspace_scheduler_mag is not None:
                self.kspace_scheduler_mag.step()
                self.kspace_scheduler_phase.step()
        elif self.parameters['kspace_architecture'] == 'KLSTM2':
            if self.kspace_scheduler is not None:
                self.kspace_scheduler.step()

        if self.ispace_scheduler is not None:
            self.ispace_scheduler.step()
        # if print_loss:
        #     print('Train Mag Loss for Epoch {} = {}' .format(epoch, avglossmag), flush = True)
        #     print('Train Phase Loss for Epoch {} = {}' .format(epoch, avglossphase), flush = True)
        #     print('Train Real Loss for Epoch {} = {}' .format(epoch, avglossreal), flush = True)
        #     print('Train SSIM for Epoch {} = {}' .format(epoch, ssim_score), flush = True)
        return avgkspacelossmag, avgkspacelossphase, avgkspacelossreal, kspacessim_score, avgkspace_l1_loss, avgkspace_l2_loss, avgispacelossreal, ispacessim_score, avgispace_l1_loss, avgispace_l2_loss

    def evaluate(self, epoch, train = False, print_loss = False):
        avgkspacelossphase = 0.
        avgkspacelossreal = 0.
        avgkspacelossmag = 0.
        kspacessim_score = 0.
        avgkspace_l1_loss = 0.
        avgkspace_l2_loss = 0.
        avgispacelossreal = 0.
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
            f = open(os.path.join(self.save_path, '{}_results.csv'.format(dstr)), 'w')
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
                undersampled_fts = torch.fft.fftshift(torch.fft.fft2(coilwise_input.to(self.device)), dim = (-2,-1))
                og_coiled_vids = og_video.to(self.device) * coils_used.to(self.device)
                og_coiled_fts = torch.fft.fftshift(torch.fft.fft2(og_coiled_vids), dim = (-2,-1))

                batch, num_frames, chan, numr, numc = undersampled_fts.shape
                inpt_mag_log = (og_coiled_fts.abs()+EPS).log()
                inpt_phase = og_coiled_fts / inpt_mag_log.exp()
                inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)
                self.kspace_model.eval()
                predr, _, _, loss_mag, loss_phase, loss_real, (loss_l1, loss_l2, ss1) = self.kspace_model(undersampled_fts, masks, self.device, periods.clone(), targ_phase = inpt_phase, targ_mag_log = inpt_mag_log, targ_real = og_coiled_vids, og_video = og_video)

                avgkspacelossphase += float(loss_phase.item()/(len(dloader)))
                avgkspacelossmag += float(loss_mag.item()/(len(dloader)))
                avgkspacelossreal += float(loss_real.item()/(len(dloader)))
                
                kspacessim_score += float(ss1.cpu()/dset.total_unskipped_frames)
                # print(kspacessim_score)
                # print(kspacessim_score)
                avgkspace_l1_loss += float(loss_l1.cpu()/dset.total_unskipped_frames)
                avgkspace_l2_loss += float(loss_l2.cpu()/dset.total_unskipped_frames)


                predr = predr.detach()[:,self.parameters['init_skip_frames']:]
                num_frames = num_frames - self.parameters['init_skip_frames']
                if self.parameters['kspace_coil_combination'] or self.parameters['ispace_lstm']:
                    predr = predr.reshape(batch*num_frames,1,numr, numc).to(self.device)
                else:
                    predr = predr.reshape(batch*num_frames,chan,numr, numc).to(self.device)
                targ_vid = og_video[:,self.parameters['init_skip_frames']:].reshape(batch*num_frames,1, numr, numc).to(self.device)

                self.ispace_model.eval()
                outp = self.ispace_model(predr)
                loss = self.l1loss(outp, targ_vid)

                if self.args.numbers_crop:
                    outp = outp[:,:,96:160,96:160]
                    targ_vid = targ_vid[:,:,96:160,96:160]

                if self.args.motion_mask:
                    motion_mask = torch.diff(targ_vid.reshape(batch, num_frames,-1)[:,np.arange(num_frames+1)%(num_frames)], n = 1, dim = 1).reshape(batch*num_frames,1,numr, numc).to(self.device)
                    motion_mask_min = motion_mask.min(1)[0].min(1)[0].min(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    motion_mask_max = motion_mask.max(1)[0].max(1)[0].max(1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    motion_mask = (motion_mask - motion_mask_min)/(motion_mask_max+EPS)
                else:
                    motion_mask = torch.ones(targ_vid.shape).to(self.device)

                loss_l1 = ((outp*motion_mask)- (targ_vid*motion_mask)).reshape(outp.shape[0]*outp.shape[1], outp.shape[2]*outp.shape[3]).abs().detach().cpu().mean(1)
                loss_l2 = ((((outp*motion_mask)- (targ_vid*motion_mask)).reshape(outp.shape[0]*outp.shape[1], outp.shape[2]*outp.shape[3]) ** 2)).detach().cpu().mean(1)
                ss1 = self.SSIM((outp*motion_mask).reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]), (targ_vid*motion_mask).reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]))
                ss1 = ss1.reshape(ss1.shape[0],-1)
                if self.args.write_csv:
                    for bi in range(batch):
                        for fi in range(num_frames):
                            f.write('{},'.format(indices[bi,0]))
                            f.write('{},'.format(indices[bi,1]))
                            f.write('{},'.format(fi))
                            f.write('{},'.format(ss1.reshape(batch, num_frames,-1).mean(2)[bi,fi]))
                            f.write('{},'.format(loss_l1.reshape(batch, num_frames)[bi,fi]))
                            f.write('{},'.format(loss_l2.reshape(batch, num_frames)[bi,fi]))
                            f.write('\n')
                            f.flush()
                loss_ss1 = ss1.mean(1).sum().detach().cpu()
                loss_l1 = loss_l1.sum()
                loss_l2 = loss_l2.sum()

                avgispacelossreal += float(loss.cpu().item()/(len(dloader)))
                ispacessim_score += float(loss_ss1.cpu().item()/(dset.total_unskipped_frames/8))
                avgispace_l1_loss += float(loss_l1.cpu().item()/(dset.total_unskipped_frames/8))
                avgispace_l2_loss += float(loss_l2.cpu().item()/(dset.total_unskipped_frames/8))

        # if print_loss:
        #     print('Train Mag Loss for Epoch {} = {}' .format(epoch, avglossmag), flush = True)
        #     print('Train Phase Loss for Epoch {} = {}' .format(epoch, avglossphase), flush = True)
        #     print('Train Real Loss for Epoch {} = {}' .format(epoch, avglossreal), flush = True)
        #     print('Train SSIM for Epoch {} = {}' .format(epoch, ssim_score), flush = True)

        # print(avgkspacelossmag, avgkspacelossphase, avgkspacelossreal, kspacessim_score, avgkspace_l1_loss, avgkspace_l2_loss, avgispacelossreal, ispacessim_score, avgispace_l1_loss, avgispace_l2_loss)
        if self.args.write_csv:
            f.flush()
            f.close()

        return avgkspacelossmag, avgkspacelossphase, avgkspacelossreal, kspacessim_score, avgkspace_l1_loss, avgkspace_l2_loss, avgispacelossreal, ispacessim_score, avgispace_l1_loss, avgispace_l2_loss

    def time_analysis(self):
        tqdm_object = tqdm(enumerate(self.testloader), total = len(self.testloader))
        times = []
        with torch.no_grad():
            for i, (indices, masks, og_video, coilwise_input, coils_used, periods) in tqdm_object:
            # for i, (indices, undersampled_fts, masks, og_coiled_fts, og_coiled_vids, og_video, periods) in tqdm_object:
                undersampled_fts = torch.fft.fftshift(torch.fft.fft2(coilwise_input.to(self.device)), dim = (-2,-1)).cpu()
                self.kspace_model.eval()
                times += self.kspace_model.module.time_analysis(undersampled_fts, self.device, periods[0:1].clone(), self.ispace_model)
                # print(time1)
                # torch.cuda.empty_cache()
                # # time2 = self.kspace_model.module.time_analysis(undersampled_fts[0:1], self.device, periods[0:1].clone(), self.ispace_model)
                # time1 = self.kspace_model.module.time_analysis(undersampled_fts[0:1], self.device, periods[0:1].clone(), self.ispace_model)
                # print(time1)
                # print(undersampled_fts.shape)
                # asdf
                # Nf, Nc, R, C = predr.squeeze().shape
                # time2 = self.ispace_model.module.time_analysis(predr.reshape(Nf,Nc,R,C).to(self.device), self.device)
                # print("Time for Complete Video ({} frames) = {}".format(undersampled_fts.shape[1], time1))
                # print("Avg Time per frame = {}".format((time1)/undersampled_fts.shape[1]))
                # print((time1)/undersampled_fts.shape[1])
                # print('\n')
                # print("Time for Ispace Video = {}".format(time2))
                # print("Time for complete video ({} frames) = {}".format(undersampled_fts.shape[1], time1+time2))
                # print("Avg Time per frame = {}".format((time1+time2)/undersampled_fts.shape[1]))
        
        # print('Average Time ({} frames) = {} +- {}'.format(undersampled_fts.shape[1], np.mean(total_times), np.std(total_times)))
        print('Average Time Per Frame = {} +- {}'.format(np.mean(times), np.std(times)), flush = True)
        scipy.io.savemat(os.path.join(self.save_path, 'fps.mat'), {'times': times})
        return


    def visualise(self, epoch, train = False):

        if self.parameters['kspace_coil_combination']:
            self.visualise_kspace(epoch, train = train)
            return
        if train:
            dloader = self.traintestloader
            dset = self.trainset
            dstr = 'Train'
            path = os.path.join(self.save_path, './images/train')
        else:
            dloader = self.testloader
            dset = self.testset
            dstr = 'Test'
            path = os.path.join(self.save_path, './images/test')
        
        print('Saving plots for {} data'.format(dstr), flush = True)
        with torch.no_grad():
            for i, (indices, masks, og_video, coilwise_input, coils_used, periods) in enumerate(dloader):
            # for i, (indices, undersampled_fts, masks, og_coiled_fts, og_coiled_vids, og_video, periods) in tqdm_object:
                undersampled_fts = torch.fft.fftshift(torch.fft.fft2(coilwise_input.to(self.device)), dim = (-2,-1))
                og_coiled_vids = og_video.to(self.device) * coils_used.to(self.device)
                og_coiled_fts = torch.fft.fftshift(torch.fft.fft2(og_coiled_vids), dim = (-2,-1))
                self.kspace_model.eval()
                self.ispace_model.eval()

                batch, num_frames, num_coils, numr, numc = undersampled_fts.shape
                
                inpt_mag_log = (og_coiled_fts.abs()+EPS).log()
                inpt_phase = og_coiled_fts / inpt_mag_log.exp()
                inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)

                tot_vids_per_patient = (dset.num_vids_per_patient*dset.frames_per_vid_per_patient)
                num_vids = 1
                batch = num_vids
                num_plots = num_vids*num_frames
                predr, _, _, loss_mag, loss_phase, loss_real, (_,_,_) = self.kspace_model(undersampled_fts[:num_vids], masks[:num_vids], self.device, periods[:num_vids].clone(), targ_phase = None, targ_mag_log = None, targ_real = None, og_video = og_video)

                print('Predr nan',torch.isnan(predr).any())
                predr[torch.isnan(predr)] = 0

                if self.parameters['ispace_lstm']:
                    predr = predr.reshape(batch*num_frames,1,numr, numc).to(self.device)
                else:
                    predr = predr.reshape(batch*num_frames,num_coils,numr, numc).to(self.device)
                targ_vid = og_video[:num_vids].reshape(batch*num_frames,1, numr, numc).to(self.device)
                ispace_outp = self.ispace_model(predr).cpu().reshape(batch,num_frames,numr,numc)
                print('ispace_outp nan',torch.isnan(ispace_outp).any())

                if self.parameters['ispace_lstm']:
                    predr = predr.reshape(batch,num_frames,1,numr,numc).repeat(1,1,num_coils,1,1)
                else:
                    predr = predr.reshape(batch,num_frames,num_coils,numr,numc)
                pred_ft = torch.fft.fftshift(torch.fft.fft2(predr), dim = (-2,-1))
                
                tot = 0
                with tqdm(total=num_plots) as pbar:
                    for bi in range(num_vids):
                        p_num, v_num = indices[bi]
                        for f_num in range(undersampled_fts.shape[1]):

                            os.makedirs(os.path.join(path, './patient_{}/by_location_number/location_{}'.format(p_num, v_num)), exist_ok=True)
                            os.makedirs(os.path.join(path, './patient_{}/by_frame_number/frame_{}'.format(p_num, f_num)), exist_ok=True)
                            
                            fig = plt.figure(figsize = (12,6))
                            og_vidi = og_video.cpu()[bi, f_num,0,:,:]
                            ispace_outpi = ispace_outp[bi, f_num, :,:]
                            
                            plt.subplot(1,3,1)
                            myimshow(og_vidi, cmap = 'gray')
                            plt.title('Ground Truth Frame')
                            # print(f_num, 1)
                            # ispace_outpi = torch.from_numpy(match_histograms(ispace_outpi.numpy(), og_vidi.numpy(), channel_axis=None))
                            plt.subplot(1,3,2)
                            myimshow(ispace_outpi, cmap = 'gray')
                            plt.title('convLSTM Prediction')
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
                            if f_num < self.parameters['init_skip_frames']:
                                spec = 'Loss Skipped'
                            plt.suptitle("Epoch {}\nPatient {} Video {} Frame {}\n{}".format(epoch,p_num, v_num, f_num, spec))
                            plt.savefig(os.path.join(path, './patient_{}/by_location_number/location_{}/io_frame_{}.jpg'.format(p_num, v_num, f_num)))
                            plt.savefig(os.path.join(path, './patient_{}/by_frame_number/frame_{}/io_location_{}.jpg'.format(p_num, f_num, v_num)))

                            if not self.args.ispace_visual_only:
                                fig = plt.figure(figsize = (36,4*num_coils-2))
                                
                                plt.subplot(num_coils,9,(((num_coils//2)-1)*9)+1)
                                myimshow(og_vidi, cmap = 'gray')
                                plt.title('Input Video')
                                # print(f_num, 1)
                                

                                plt.subplot(num_coils,9,(((num_coils//2))*9))
                                myimshow(ispace_outpi, cmap = 'gray')
                                plt.title('Ispace Prediction')
                                # print(f_num, 2)
                                
                                plt.subplot(num_coils,9,(((num_coils//2)+1)*9))
                                diffvals = show_difference_image(ispace_outpi, og_vidi.numpy())
                                plt.title('Difference Frame')
                                # print(f_num, 3)

                                # plt.subplot(8,8,(((num_coils//2)+2)*8))
                                # plt.hist(diffvals, range = [0,0.25], density = False, bins = 30)
                                # plt.ylabel('Pixel Count')
                                # plt.xlabel('Difference Value')
                                # print(f_num, 4)

                                num_plots -= 1
                                for c_num in range(num_coils):
                                    if num_plots == 0:
                                        return
                                    
                                    targi = og_coiled_vids.cpu()[bi,f_num, c_num].squeeze().cpu().numpy()
                                    orig_fti = (og_coiled_fts.cpu()[bi,f_num,c_num].abs()+1).log()
                                    mask_fti = (1+undersampled_fts.cpu()[bi,f_num,c_num].abs()).log()
                                    ifft_of_undersamp = torch.fft.ifft2(torch.fft.ifftshift(undersampled_fts.cpu()[bi,f_num,c_num], dim = (-2,-1))).abs().squeeze()
                                    pred_fti = (pred_ft.cpu()[bi,f_num,c_num].abs()+1).log()
                                    predi = predr.cpu()[bi,f_num,c_num].squeeze().cpu().numpy()

                                    plt.subplot(num_coils,9,9*c_num+2)
                                    myimshow(targi, cmap = 'gray')
                                    if c_num == 0:
                                        plt.title('Coiled Input')
                                    # print(c_num,1)
                                    
                                    plt.subplot(num_coils,9,9*c_num+3)
                                    myimshow((orig_fti).squeeze(), cmap = 'gray')
                                    if c_num == 0:
                                        plt.title('FT of Coiled Input')
                                    # print(c_num,2)
                                    
                                    plt.subplot(num_coils,9,9*c_num+4)
                                    myimshow(mask_fti, cmap = 'gray')
                                    if c_num == 0:
                                        plt.title('Undersampled FT')
                                    # print(c_num,3)
                                    
                                    plt.subplot(num_coils,9,9*c_num+5)
                                    myimshow(ifft_of_undersamp, cmap = 'gray')
                                    if c_num == 0:
                                        plt.title('IFFT of Undersampled FT')
                                    # print(c_num,4)
                                    
                                    plt.subplot(num_coils,9,9*c_num+6)
                                    myimshow(pred_fti, cmap = 'gray')
                                    if c_num == 0:
                                        plt.title('FT predicted by Kspace Model')
                                    # print(c_num,5)
                                    
                                    plt.subplot(num_coils,9,9*c_num+7)
                                    myimshow(predi, cmap = 'gray')
                                    if c_num == 0:
                                        plt.title('IFFT of Kspace Prediction')
                                    # print(c_num,6)
                                    
                                    plt.subplot(num_coils,9,9*c_num+8)
                                    diffvals = show_difference_image(predi, targi)
                                    if c_num == 0:
                                        plt.title('Difference Image')
                                    # print(c_num,6)
                                    
                                spec = ''
                                if f_num < self.parameters['init_skip_frames']:
                                    spec = 'Loss Skipped'
                                plt.suptitle("Epoch {}\nPatient {} Video {} Frame {}\n{}".format(epoch,p_num, v_num, f_num, spec))
                                plt.savefig(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}.jpg'.format(p_num, v_num, f_num)))
                                plt.savefig(os.path.join(path, './patient_{}/by_frame_number/frame_{}/location_{}.jpg'.format(p_num, f_num, v_num)))
                            plt.close('all')

                            tot += 1
                            pbar.update(1)
                break
