import os
import gc
import sys
import PIL
import time
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

EPS = 1e-10
CEPS = torch.complex(torch.tensor(EPS),torch.tensor(EPS))

import sys
sys.path.append('/root/Cardiac-MRI-Reconstrucion/')

from utils.functions import fetch_loss_function

def myimshow(x, cmap = 'gray', trim = False):
    if trim:
        percentile_95 = np.percentile(x, 95)
        percentile_5 = np.percentile(x, 5)
        x[x > percentile_95] = percentile_95
        x[x < percentile_5] = percentile_5
    x = x - x.min()
    x = x/ (x.max() + EPS)
    plt.imshow(x, cmap = cmap)

def special_trim(x):
    percentile_95 = np.percentile(x.detach().cpu(), 95)
    percentile_5 = np.percentile(x.detach().cpu(), 5)
    x[x > percentile_95] = percentile_95
    x[x < percentile_5] = percentile_5
    x = x - x.min().detach()
    x = x/ (x.max().detach() + EPS)
    return x

def show_difference_image(im1, im2):
    im1 = (im1 - im1.min())
    im1 = (im1 / im1.max())
    im2 = (im2 - im2.min())
    im2 = (im2 / im2.max())
    diff = (im1-im2)
    plt.imshow(np.abs(diff), cmap = 'plasma', vmin=0, vmax=0.25)
    plt.colorbar()
    return np.abs(diff).reshape(-1)

# takes FT, FT_mask
# num_coils, num_window, 256, 256, 2
def input_save(fts, masks, targets, path):
    num_coils = fts.shape[0]
    num_windows = fts.shape[1]
    avg_FT = fts.mean(1)
    avg_FT = torch.complex(avg_FT[:,:,0], avg_FT[:,:,1])
    combined_ft_undersampled = avg_FT.clone()*0
    averager = torch.zeros(combined_ft_undersampled.shape)
    for wi in range(num_windows):
        mask_curr = masks[:,wi]
        ft_curr = fts_masked[:,wi] * mask_curr
        x = torch.complex(ft_curr[:,:,:,0],ft_curr[:,:,:,1])
        combined_ft_undersampled += x
        averager[x != 0] += 1
    combined_ft_undersampled[averager != 0] /= averager[averager != 0]

    for coili in range(num_coils):
        fig = plt.figure(figsize = (28,16))
        iter = 1
        for wi in range(num_windows):
            plt.subplot(4,num_windows,iter)
            ft = torch.complex(fts[coili,wi,:,:,0],fts[coili,wi,:,:,1])
            myimshow((ft+CEPS).log().abs(), cmap = 'gray')
            if wi == 3:
                plt.title('Complete FFT')
            iter += 1
        for wi in range(num_windows):
            plt.subplot(4,num_windows,iter)
            ft = torch.complex(fts[coili,wi,:,:,0],fts[coili,wi,:,:,1])
            outp = torch.fft.ifft2(torch.fft.ifftshift(ft, dim = (-2, -1))).real
            myimshow(outp, cmap = 'gray')
            if wi == 3:
                plt.title('Original Image')
            iter += 1
        for wi in range(num_windows):
            plt.subplot(4,num_windows,iter)
            mask_curr = masks[coili,wi]
            ft_curr = fts_masked[coili,wi] * mask_curr
            ft = torch.complex(ft_curr[:,:,0],ft_curr[:,:,1])
            myimshow((ft+CEPS).log().abs(), cmap = 'gray')
            if wi == 3:
                plt.title('Undersampled FFT')
            iter += 1
        for wi in range(num_windows):
            plt.subplot(4,num_windows,iter)
            mask_curr = masks[coili,wi]
            ft_curr = fts_masked[coili,wi] * mask_curr
            ft = torch.complex(ft_curr[:,:,0],ft_curr[:,:,1])
            outp = torch.fft.ifft2(torch.fft.ifftshift(ft, dim = (-2, -1))).real
            myimshow(outp, cmap = 'gray')
            if wi == 3:
                plt.title('Partial Image')
            iter += 1
        plt.suptitle("Coil {}".format(coili))
        plt.savefig(os.path.join(path, 'coil_{}.png'.format(coili)))
        plt.close('all')
        fig = plt.figure(figsize = (8,8))
        plt.subplot(2,2,1)
        myimshow(avg_FT[coili,:,:].abs(), cmap = 'gray')
        plt.title('Averaged Complete FTs')
        plt.subplot(2,2,2)
        ft = avg_FT[coili,:,:]
        outp = torch.fft.ifft2(torch.fft.ifftshift(ft.exp(), dim = (-2, -1))).real
        myimshow(outp, cmap = 'gray')
        plt.title('Complete FT - Image')
        plt.subplot(2,2,3)
        myimshow(combined_ft_undersampled[coili,:,:].abs(), cmap = 'gray')
        plt.title('Averaged Complete FTs')
        plt.subplot(2,2,4)
        ft = combined_ft_undersampled[coili,:,:]
        outp = torch.fft.ifft2(torch.fft.ifftshift(ft.exp(), dim = (-2, -1))).real
        myimshow(outp, cmap = 'gray')
        plt.title('Undersampled FT - Image')
        plt.savefig(os.path.join(path, 'combined_coil_{}.png'.format(coili)))
        plt.close('all')
    fig = plt.figure(figsize = (4,4))
    myimshow(targets.squeeze().numpy(), cmap = 'gray')
    plt.title('Target')
    plt.savefig(os.path.join(path, 'target.png'))
    plt.close('all')

        







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
                            pin_memory = True,
                            drop_last = False,
                            sampler = self.train_sampler
                        )
        self.traintestloader = torch.utils.data.DataLoader(
                            self.trainset,
                            batch_size=self.parameters['test_batch_size'], 
                            shuffle = False,
                            num_workers = self.parameters['dataloader_num_workers'],
                            pin_memory = True,
                            drop_last = False,
                            sampler = self.train_test_sampler
                        )
        self.testloader = torch.utils.data.DataLoader(
                            self.testset,
                            batch_size=self.parameters['test_batch_size'], 
                            shuffle = False,
                            num_workers = self.parameters['dataloader_num_workers'],
                            pin_memory = True,
                            drop_last = False,
                            sampler = self.test_sampler
                        )

        if self.parameters['optimizer'] == 'Adam':
            self.kspace_optim = optim.Adam(self.kspace_model.parameters(), lr=self.parameters['lr_kspace'], betas=self.parameters['optimizer_params'])
            self.ispace_optim = optim.Adam(self.ispace_model.parameters(), lr=self.parameters['lr_ispace'], betas=self.parameters['optimizer_params'])
            self.parameters['scheduler_params']['cycle_momentum'] = False
            
        if self.parameters['scheduler'] == 'None':
            self.kspace_scheduler = None
            self.ispace_scheduler = None
        if self.parameters['scheduler'] == 'StepLR':
            mydic = self.parameters['scheduler_params']
            if self.ddp_rank == 0:
                self.kspace_scheduler = optim.lr_scheduler.StepLR(self.kspace_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=mydic['verbose'])
                self.ispace_scheduler = optim.lr_scheduler.StepLR(self.ispace_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=mydic['verbose'])
            else:
                self.kspace_scheduler = optim.lr_scheduler.StepLR(self.kspace_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=False)
                self.ispace_scheduler = optim.lr_scheduler.StepLR(self.ispace_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=False)
        if self.parameters['scheduler'] == 'CyclicLR':
            mydic = self.parameters['scheduler_params']
            if self.ddp_rank == 0:
                self.kspace_scheduler = optim.lr_scheduler.CyclicLR(self.kspace_optim, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'],  verbose=mydic['verbose'])
                self.ispace_scheduler = optim.lr_scheduler.CyclicLR(self.ispace_optim, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'],  verbose=mydic['verbose'])
            else:
                self.kspace_scheduler = optim.lr_scheduler.CyclicLR(self.kspace_optim, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'])
                self.ispace_scheduler = optim.lr_scheduler.CyclicLR(self.ispace_optim, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'])

        self.l1loss = fetch_loss_function('L1',self.device, self.parameters['loss_params'])
        self.l2loss = fetch_loss_function('L2',self.device, self.parameters['loss_params'])
        self.SSIM = kornia.metrics.SSIM(11)
        # if self.criterion_FT is not None:
        #     self.criterion_FT = self.criterion_FT.to(self.device)
        # if self.criterion_reconFT is not None:
        #     self.criterion_reconFT = self.criterion_reconFT.to(self.device)

    def train(self, epoch, print_loss = False):
        if epoch >= self.parameters['num_epochs_kspace']:
            return self.train_ispace(epoch, print_loss)
        else:
            return self.train_kspace(epoch, print_loss)

    def evaluate(self, epoch, train = False, print_loss = False):
        if epoch >= self.parameters['num_epochs_kspace']:
            return self.evaluate_ispace(epoch, train, print_loss)
        else:
            return self.evaluate_kspace(epoch, train, print_loss)

    def train_ispace(self, epoch, print_loss = False):
        pass

    def train_kspace(self, epoch, print_loss = False):
        avglossphase = 0
        avglossreal = 0
        avglossmag = 0
        ssim_score = 0
        avg_l1_loss = 0
        avg_l2_loss = 0
        self.trainloader.sampler.set_epoch(epoch)
        if self.ddp_rank == 0:
            tqdm_object = tqdm(enumerate(self.trainloader), total = len(self.trainloader), desc = "[{}] | Epoch {}".format(os.getpid(), epoch), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
        else:
            tqdm_object = enumerate(self.trainloader)
        for i, (indices, fts, masks, targets, target_fts, periods) in tqdm_object:
            self.kspace_optim.zero_grad(set_to_none=True)
            # with autocast(enabled = self.parameters['Automatic_Mixed_Precision'], dtype=torch.float32):
            # with autocast(enabled = self.parameters['Automatic_Mixed_Precision']):
            # self.kspace_model.module.train_mode_set(True)
            
            # pred1 = torch.fft.ifft2(torch.fft.ifftshift(fts, dim = (-2,-1))).real
            # pred2 = torch.fft.ifft2(torch.fft.ifftshift(target_fts, dim = (-2,-1))).real
            # for i in range(10):
            #     plt.imsave('pred1_{}.png'.format(i), pred1[0,i,0,:,:])
            #     plt.imsave('pred2_{}.png'.format(i), pred2[0,i,:,:])
            # asdf
            # if i == 10:
            #     asdf

            batch, num_frames, chan, numr, numc = fts.shape
            inpt_mag_log = (target_fts+CEPS).log().real
            inpt_phase = target_fts / inpt_mag_log.exp()
            inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)
            self.kspace_model.train()
            predr, ans_phase, ans_mag_log, loss_mag, loss_phase, loss_real = self.kspace_model(fts,masks, self.device, periods.clone(), targ_phase = inpt_phase, targ_mag_log = inpt_mag_log, targ_real = targets)
            # loss = loss_mag + 8*loss_phase + loss_real
            loss = loss_mag + 8*loss_phase
            loss.backward()
            self.kspace_optim.step()

            avglossphase += loss_phase.item()/(len(self.trainloader))
            avglossmag += loss_mag.item()/(len(self.trainloader))
            avglossreal += loss_real.item()/(len(self.trainloader))

            
            ss1 = self.SSIM(predr.to(self.device).reshape(batch*num_frames,1,numr,numc), targets.reshape(batch*num_frames,1,numr,numc).to(self.device))
            ss1 = ss1.reshape(ss1.shape[0], -1).mean(1)
            ssim_score += (ss1/(num_frames)).sum().item()/len(self.trainset)
            avg_l1_loss += self.l1loss(predr, targets)/len(self.trainset)
            avg_l2_loss += self.l2loss(predr, targets)/len(self.trainset)

        if self.kspace_scheduler is not None:
            self.kspace_scheduler.step()
        if print_loss:
            print('Train Mag Loss for Epoch {} = {}' .format(epoch, avglossmag), flush = True)
            print('Train Phase Loss for Epoch {} = {}' .format(epoch, avglossphase), flush = True)
            print('Train Real Loss for Epoch {} = {}' .format(epoch, avglossreal), flush = True)
            print('Train SSIM for Epoch {} = {}' .format(epoch, ssim_score), flush = True)

        return avglossmag, avglossphase, avglossreal, ssim_score, avg_l1_loss, avg_l2_loss

    def evaluate_kspace(self, epoch, train = False, print_loss = False):
        avglossphase = 0
        avglossreal = 0
        avglossmag = 0
        ssim_score = 0
        avg_l1_loss = 0
        avg_l2_loss = 0
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
        if self.ddp_rank == 0:
            tqdm_object = tqdm(enumerate(dloader), total = len(dloader), desc = "Testing after Epoch {} on {}set".format(epoch, dstr), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
        else:
            tqdm_object = enumerate(self.trainloader)
        with torch.no_grad():
            for i, (indices, fts, masks, targets, target_fts, periods) in tqdm_object:
                # with autocast(enabled = self.parameters['Automatic_Mixed_Precision'], dtype=torch.float32):
                # with autocast(enabled = self.parameters['Automatic_Mixed_Precision']):
                # self.kspace_model.module.train_mode_set(True)
                batch, num_frames, chan, numr, numc = fts.shape
                inpt_mag_log = (target_fts+CEPS).log().real
                inpt_phase = target_fts / inpt_mag_log.exp()
                inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)
                self.kspace_model.eval()

                predr, ans_phase, ans_mag_log, loss_mag, loss_phase, loss_real = self.kspace_model(fts,masks, self.device, periods.clone(), targ_phase = inpt_phase, targ_mag_log = inpt_mag_log, targ_real = targets)

                avglossphase += loss_phase.item()/(len(self.trainloader))
                avglossmag += loss_mag.item()/(len(self.trainloader))
                avglossreal += loss_real.item()/(len(self.trainloader))

                ss1 = self.SSIM(predr.to(self.device).reshape(batch*num_frames,1,numr,numc), targets.reshape(batch*num_frames,1,numr,numc).to(self.device))
                ss1 = ss1.reshape(ss1.shape[0], -1).mean(1)
                ssim_score += (ss1/(num_frames)).sum().item()/len(dset)
                avg_l1_loss += self.l1loss(predr, targets)/len(dset)
                avg_l2_loss += self.l2loss(predr, targets)/len(dset)

            if self.kspace_scheduler is not None:
                self.kspace_scheduler.step()
            if print_loss:
                print('Train Mag Loss for Epoch {} = {}' .format(epoch, avglossmag), flush = True)
                print('Train Phase Loss for Epoch {} = {}' .format(epoch, avglossphase), flush = True)
                print('Train Real Loss for Epoch {} = {}' .format(epoch, avglossreal), flush = True)
                print('Train SSIM for Epoch {} = {}' .format(epoch, ssim_score), flush = True)

            return avglossmag, avglossphase, avglossreal, ssim_score, avg_l1_loss, avg_l2_loss

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
        tot_vids_per_patient = (dset.num_vids_per_patient*dset.frames_per_vid_per_patient)
        num_plots = tot_vids_per_patient[0]
        num_plots = 30
        print('Saving plots for {} data'.format(dstr), flush = True)
        with torch.no_grad():
            for i, (indices, fts, masks, targets, target_fts, periods) in tqdm(enumerate(dloader), total = 1+num_plots//self.parameters['test_batch_size']):
                if not os.path.exists(os.path.join(self.args.run_id, './images/input/')):
                    os.mkdir(os.path.join(self.args.run_id, './images/input/'))
                if not os.path.exists(os.path.join(self.args.run_id, './images/input2/')):
                    os.mkdir(os.path.join(self.args.run_id, './images/input2/'))
                # input_save(fts[0], masks[0], targets[0], os.path.join(self.args.run_id, './images/input/'))
                # input_save(fts[1], masks[1], targets[1], os.path.join(self.args.run_id, './images/input2/'))
                # self.kspace_model.module.train_mode_set(False)
                self.kspace_model.eval()
                batch, num_frames, chan, numr, numc = fts.shape
                inpt_mag_log = (target_fts+CEPS).log().real
                inpt_phase = target_fts / inpt_mag_log.exp()
                inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)

                predr, ans_phase, ans_mag_log, loss_mag, loss_phase, loss_real = self.kspace_model(fts,masks, self.device, periods.clone(), targ_phase = inpt_phase, targ_mag_log = inpt_mag_log, targ_real = targets)

                with torch.no_grad():
                    pred_ft = ans_phase*ans_mag_log.unsqueeze(-1).exp()
                    pred_ft = torch.complex(pred_ft[:,:,:,:,0],pred_ft[:,:,:,:,1])
                    predr = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).real

                for bi in range(fts.shape[0]):
                    for f_num in range(fts.shape[1]):
                        if num_plots == 0:
                            return
                        num_plots -= 1
                        targi = targets[bi,f_num].squeeze().cpu().numpy()
                        predi = predr[bi,f_num].squeeze().cpu().numpy()
                        p_num, v_num, _ = indices[bi]
                        orig_fti = (target_fts[bi,f_num]+CEPS).log().real
                        maski = masks[bi,f_num]
                        pred_fti = (pred_ft[bi,f_num]+CEPS).log().real

                        temp = torch.complex(target_fts[bi,f_num].real*maski,target_fts[bi,f_num].imag*maski)
                        ft_of_undersamp = torch.fft.ifft2(torch.fft.ifftshift(temp, dim = (-2,-1))).real.squeeze()

                        # fig = plt.figure(figsize = (8,8))
                        # plt.subplot(2,2,1)
                        # ft = torch.complex(fts_masked[i,0,3,:,:,0],fts_masked[i,0,3,:,:,1])
                        # myimshow(ft.abs(), cmap = 'gray')
                        # plt.title('Undersampled FFT Frame')
                        # plt.subplot(2,2,2)
                        # myimshow(torch.fft.ifft2(torch.fft.ifftshift(ft.exp())).real, cmap = 'gray')
                        # plt.title('IFFT of the Input')
                        # plt.subplot(2,2,3)
                        # myimshow(predi, cmap = 'gray')
                        # plt.title('Our Predicted Frame')
                        # plt.subplot(2,2,4)
                        # myimshow(targi, cmap = 'gray')
                        # plt.title('Actual Frame')
                        # plt.suptitle("{} data window index {}".format(dstr, indices[i]))
                        # plt.savefig(os.path.join(path, '{}_result_epoch{}_{}'.format(dstr, epoch, i)))
                        # plt.close('all')
                        fig = plt.figure(figsize = (16,8))
                        # plt.subplot(2,2,1)
                        # ft = torch.complex(fts_masked[i,0,3,:,:,0],fts_masked[i,0,3,:,:,1])
                        # myimshow(ft.abs(), cmap = 'gray')
                        # plt.title('Undersampled FFT Frame')
                        # plt.subplot(2,2,2)
                        # myimshow(torch.fft.ifft2(torch.fft.ifftshift(ft.exp())).real, cmap = 'gray')
                        # plt.title('IFFT of the Input')
                        plt.subplot(2,4,1)
                        myimshow(orig_fti, cmap = 'gray')
                        plt.title('Original FFT')
                        plt.subplot(2,4,2)
                        myimshow((orig_fti*maski).squeeze(), cmap = 'gray')
                        plt.title('Masked FFT')
                        plt.subplot(2,4,3)
                        myimshow(pred_fti, cmap = 'gray')
                        plt.title('Predicted FFT')
                        plt.subplot(2,4,4)
                        diffvals = show_difference_image(predi, targi)
                        plt.title('Difference Frame')
                        plt.subplot(2,4,5)
                        myimshow(targi, cmap = 'gray')
                        plt.title('Actual Frame')
                        plt.subplot(2,4,6)
                        myimshow(ft_of_undersamp, cmap = 'gray')
                        plt.title('IFFT of undersampled data')
                        plt.subplot(2,4,7)
                        myimshow(predi, cmap = 'gray',trim = True)
                        plt.title('Our Predicted Frame')
                        plt.subplot(2,4,8)
                        plt.hist(diffvals, range = [0,0.25], density = False, bins = 30)
                        plt.ylabel('Pixel Count')
                        plt.xlabel('Difference Value')
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
                        plt.suptitle("Patient {} Location {} Frame {}".format(p_num, v_num, f_num))
                        plt.savefig(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}.jpg'.format(p_num, v_num, f_num)))
                        plt.savefig(os.path.join(path, './patient_{}/by_frame_number/frame_{}/location_{}.jpg'.format(p_num, f_num, v_num)))
                        plt.close('all')