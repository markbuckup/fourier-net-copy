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
    im1 = (im1 / im1.max() + EPS)
    im2 = (im2 - im2.min())
    im2 = (im2 / im2.max() + EPS)
    diff = (im1-im2)
    plt.imshow(np.abs(diff), cmap = 'plasma', vmin=0, vmax=0.25)
    plt.colorbar()
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
        # self.l2loss = fetch_loss_function('L2',self.device, self.parameters['loss_params'])
        self.SSIM = kornia.metrics.SSIM(11)
        # if self.criterion_FT is not None:
        #     self.criterion_FT = self.criterion_FT.to(self.device)
        # if self.criterion_reconFT is not None:
        #     self.criterion_reconFT = self.criterion_reconFT.to(self.device)

    def train(self, epoch, print_loss = False):
        self.k_space_trainable = True
        if epoch >= self.parameters['num_epochs_kspace']:
            if self.k_space_trainable:
                for param in self.kspace_model.parameters():
                    param.requires_grad = False
                self.k_space_trainable = False
            return self.train_ispace(epoch, print_loss)
        else:
            assert(self.k_space_trainable)
            return self.train_kspace(epoch, print_loss)

    def evaluate(self, epoch, train = False, print_loss = False):
        if epoch >= self.parameters['num_epochs_kspace']:
            return self.evaluate_ispace(epoch, train, print_loss)
        else:
            return self.evaluate_kspace(epoch, train, print_loss)

    def train_ispace(self, epoch, print_loss = False):
        avglossreal = 0
        ssim_score = 0
        avg_l1_loss = 0
        avg_l2_loss = 0
        self.trainloader.sampler.set_epoch(epoch)
        if self.ddp_rank == 0:
            tqdm_object = tqdm(enumerate(self.trainloader), total = len(self.trainloader), desc = "[{}] | IS Epoch {}".format(os.getpid(), epoch), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
        else:
            tqdm_object = enumerate(self.trainloader)
        for i, (indices, undersampled_fts, og_coiled_fts, og_coiled_vids, og_video, periods) in tqdm_object:
            with torch.no_grad():
                batch, num_frames, chan, numr, numc = undersampled_fts.shape
                inpt_mag_log = (og_coiled_fts.abs()+EPS).log()
                inpt_phase = og_coiled_fts / inpt_mag_log.exp()
                inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)
                self.kspace_model.train()
                mask = None
                predr, _, _, _, _, _, _ = self.kspace_model(undersampled_fts, mask, self.device, periods.clone(), targ_phase = None, targ_mag_log = None, targ_real = None)
            
            predr = predr[:,self.parameters['init_skip_frames']:]
            num_frames = num_frames - self.parameters['init_skip_frames']
            predr = predr.reshape(batch*num_frames,chan,numr, numc).to(self.device)
            targ_vid = og_video[:,self.parameters['init_skip_frames']:].reshape(batch*num_frames,1, numr, numc).to(self.device)
            self.ispace_optim.zero_grad(set_to_none=True)

            outp = self.ispace_model(predr)
            loss = self.l1loss(outp, targ_vid)

            loss.backward()
            self.ispace_optim.step()

            loss_l1 = (outp- targ_vid).reshape(outp.shape[0]*outp.shape[1], outp.shape[2]*outp.shape[3]).abs().mean(1).sum().detach().cpu()
            loss_l2 = (((outp- targ_vid).reshape(outp.shape[0]*outp.shape[1], outp.shape[2]*outp.shape[3]) ** 2).mean(1).sum()).detach().cpu()
            ss1 = self.SSIM(outp.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]), targ_vid.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]))
            ss1 = ss1.reshape(ss1.shape[0],-1)
            loss_ss1 = ss1.mean(1).sum().detach().cpu()

            avglossreal += loss.item()/(len(self.trainloader))
            ssim_score += loss_ss1/self.trainset.total_frames
            avg_l1_loss += loss_l1/self.trainset.total_frames
            avg_l2_loss += loss_l2/self.trainset.total_frames

        if self.ispace_scheduler is not None:
            self.ispace_scheduler.step()
        if print_loss:
            print('Train Train Loss for Epoch {} = {}' .format(epoch, avglossreal), flush = True)
            print('Train SSIM for Epoch {} = {}' .format(epoch, ssim_score), flush = True)

        return 0,0,avglossreal, ssim_score, avg_l1_loss, avg_l2_loss

    def train_kspace(self, epoch, print_loss = False):
        avglossphase = 0
        avglossreal = 0
        avglossmag = 0
        ssim_score = 0
        avg_l1_loss = 0
        avg_l2_loss = 0
        self.trainloader.sampler.set_epoch(epoch)
        if self.ddp_rank == 0:
            tqdm_object = tqdm(enumerate(self.trainloader), total = len(self.trainloader), desc = "[{}] | KS Epoch {}".format(os.getpid(), epoch), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
        else:
            tqdm_object = enumerate(self.trainloader)
        for i, (indices, undersampled_fts, og_coiled_fts, og_coiled_vids, og_video, periods) in tqdm_object:
            self.kspace_optim.zero_grad(set_to_none=True)
            
            batch, num_frames, chan, numr, numc = undersampled_fts.shape
            inpt_mag_log = (og_coiled_fts.abs()+EPS).log()
            inpt_phase = og_coiled_fts / inpt_mag_log.exp()
            inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)
            self.kspace_model.train()
            mask = None
            predr, ans_phase, ans_mag_log, loss_mag, loss_phase, loss_real, (loss_l1, loss_l2, ss1) = self.kspace_model(undersampled_fts, mask, self.device, periods.clone(), targ_phase = inpt_phase, targ_mag_log = inpt_mag_log, targ_real = og_coiled_vids)
            loss = 0.1*loss_mag + 3*loss_phase + 5*loss_real

            # loss = loss_mag + 8*loss_phase
            loss.backward()
            # print(loss)
            self.kspace_optim.step()

            avglossphase += loss_phase.item()/(len(self.trainloader))
            avglossmag += loss_mag.item()/(len(self.trainloader))
            avglossreal += loss_real.item()/(len(self.trainloader))

            
            ssim_score += ss1/self.trainset.total_frames
            avg_l1_loss += loss_l1/self.trainset.total_frames
            avg_l2_loss += loss_l2/self.trainset.total_frames

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
            for i, (indices, undersampled_fts, og_coiled_fts, og_coiled_vids, og_video, periods) in tqdm_object:
                # with autocast(enabled = self.parameters['Automatic_Mixed_Precision'], dtype=torch.float32):
                # with autocast(enabled = self.parameters['Automatic_Mixed_Precision']):
                # self.kspace_model.module.train_mode_set(True)
                batch, num_frames, chan, numr, numc = undersampled_fts.shape
                inpt_mag_log = (og_coiled_fts.abs() + EPS).log()
                inpt_phase = og_coiled_fts / inpt_mag_log.exp()
                inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)
                self.kspace_model.eval()

                mask = None
                predr, ans_phase, ans_mag_log, loss_mag, loss_phase, loss_real, (loss_l1, loss_l2, ss1) = self.kspace_model(undersampled_fts, mask, self.device, periods.clone(), targ_phase = inpt_phase, targ_mag_log = inpt_mag_log, targ_real = og_coiled_vids)

                avglossphase += loss_phase.item()/(len(self.trainloader))
                avglossmag += loss_mag.item()/(len(self.trainloader))
                avglossreal += loss_real.item()/(len(self.trainloader))

                ssim_score += ss1/dset.total_frames
                avg_l1_loss += loss_l1/dset.total_frames
                avg_l2_loss += loss_l2/dset.total_frames

            if self.kspace_scheduler is not None:
                self.kspace_scheduler.step()
            if print_loss:
                print('Train Mag Loss for Epoch {} = {}' .format(epoch, avglossmag), flush = True)
                print('Train Phase Loss for Epoch {} = {}' .format(epoch, avglossphase), flush = True)
                print('Train Real Loss for Epoch {} = {}' .format(epoch, avglossreal), flush = True)
                print('Train SSIM for Epoch {} = {}' .format(epoch, ssim_score), flush = True)

            return avglossmag, avglossphase, avglossreal, ssim_score, avg_l1_loss, avg_l2_loss

    def evaluate_ispace(self, epoch, train = False, print_loss = False):
        avglossreal = 0
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
            for i, (indices, undersampled_fts, og_coiled_fts, og_coiled_vids, og_video, periods) in tqdm_object:
                batch, num_frames, chan, numr, numc = undersampled_fts.shape
                inpt_mag_log = (og_coiled_fts.abs()+EPS).log()
                inpt_phase = og_coiled_fts / inpt_mag_log.exp()
                inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)
                self.kspace_model.train()
                mask = None
                predr, _, _, _, _, _, _ = self.kspace_model(undersampled_fts, mask, self.device, periods.clone(), targ_phase = None, targ_mag_log = None, targ_real = None)
                
                predr = predr[:,self.parameters['init_skip_frames']:]
                num_frames = num_frames - self.parameters['init_skip_frames']
                predr = predr.reshape(batch*num_frames,chan,numr, numc).to(self.device)
                targ_vid = og_video[:,self.parameters['init_skip_frames']:].reshape(batch*num_frames,1, numr, numc).to(self.device)

                outp = self.ispace_model(predr)
                loss = self.l1loss(outp, targ_vid)

                loss_l1 = (outp- targ_vid).reshape(outp.shape[0]*outp.shape[1], outp.shape[2]*outp.shape[3]).abs().mean(1).sum().detach().cpu()
                loss_l2 = (((outp- targ_vid).reshape(outp.shape[0]*outp.shape[1], outp.shape[2]*outp.shape[3]) ** 2).mean(1).sum()).detach().cpu()
                ss1 = self.SSIM(outp.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]), targ_vid.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]))
                ss1 = ss1.reshape(ss1.shape[0],-1)
                loss_ss1 = ss1.mean(1).sum().detach().cpu()

                avglossreal += loss.item()/(len(self.trainloader))
                ssim_score += loss_ss1/self.trainset.total_frames
                avg_l1_loss += loss_l1/self.trainset.total_frames
                avg_l2_loss += loss_l2/self.trainset.total_frames

            if self.ispace_scheduler is not None:
                self.ispace_scheduler.step()
            if print_loss:
                print('Train Train Loss for Epoch {} = {}' .format(epoch, avglossreal), flush = True)
                print('Train SSIM for Epoch {} = {}' .format(epoch, ssim_score), flush = True)

        return 0,0,avglossreal, ssim_score, avg_l1_loss, avg_l2_loss

    def visualise_ispace(self, epoch, train = False):
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
            for i, (indices, undersampled_fts, og_coiled_fts, og_coiled_vids, og_video, periods) in enumerate(dloader):
                if not os.path.exists(os.path.join(self.args.run_id, './images/input/')):
                    os.mkdir(os.path.join(self.args.run_id, './images/input/'))
                if not os.path.exists(os.path.join(self.args.run_id, './images/input2/')):
                    os.mkdir(os.path.join(self.args.run_id, './images/input2/'))
                # input_save(fts[0], masks[0], targets[0], os.path.join(self.args.run_id, './images/input/'))
                # input_save(fts[1], masks[1], targets[1], os.path.join(self.args.run_id, './images/input2/'))
                # self.kspace_model.module.train_mode_set(False)
                self.kspace_model.eval()
                batch, num_frames, num_coils, numr, numc = undersampled_fts.shape
                inpt_mag_log = (og_coiled_fts.abs()+EPS).log()
                inpt_phase = og_coiled_fts / inpt_mag_log.exp()
                inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)

                tot_vids_per_patient = (dset.num_vids_per_patient*dset.frames_per_vid_per_patient)
                num_vids = 1
                num_plots = sum(tot_vids_per_patient[:num_vids]*num_coils)

                mask = None
                predr, ans_phase, ans_mag_log, loss_mag, loss_phase, loss_real, (_,_,_) = self.kspace_model(undersampled_fts[:num_vids], mask, self.device, periods[:num_vids].clone(), targ_phase = inpt_phase[:num_vids], targ_mag_log = inpt_mag_log[:num_vids], targ_real = og_coiled_vids[:num_vids])

                predr = predr[:,self.parameters['init_skip_frames']:]
                batch, num_frames, num_coils, numr, numc = predr.shape
                predr = predr.reshape(batch*num_frames,num_coils,numr, numc).to(self.device)
                targ_vid = og_video[:num_vids,self.parameters['init_skip_frames']:].reshape(batch*num_frames,1, numr, numc).to(self.device)
                outp = self.ispace_model(predr).cpu()

                outp = outp.reshape(batch, num_frames,numr, numc).cpu()
                predr = predr.reshape(batch, num_frames, num_coils, numr, numc).cpu()
                targ_vid = targ_vid.reshape(batch, num_frames,numr, numc).cpu()


                for i in range(10):
                    for ci in range(8):
                        predi = outp[0,-i,:,:]
                        targi = targ_vid[0,-i,:,:]
                        kspace_outi = predr[0,-i,ci,:,:]

                        fig = plt.figure(figsize = (8,8))
                        plt.subplot(2,2,1)
                        myimshow(kspace_outi, cmap = 'gray', trim = True)
                        plt.title('Predicted By Kspace Model')
                        plt.subplot(2,2,2)
                        myimshow(predi, cmap = 'gray')
                        plt.title('Predicted By Ispace Model')
                        plt.subplot(2,2,3)
                        myimshow(targi, cmap = 'gray')
                        plt.title('Original Frame')
                        plt.subplot(2,2,4)
                        diffvals = show_difference_image(predi, targi)
                        plt.title('Difference Frame')

                        plt.suptitle("Frame {}".format(i))
                        plt.savefig(os.path.join(path, 'frame_{}_coil_{}.jpg'.format(i,ci)))
                        plt.close('all')

    def visualise_kspace(self, epoch, train = False):
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
            for i, (indices, undersampled_fts, og_coiled_fts, og_coiled_vids, og_video, periods) in enumerate(dloader):
                if not os.path.exists(os.path.join(self.args.run_id, './images/input/')):
                    os.mkdir(os.path.join(self.args.run_id, './images/input/'))
                if not os.path.exists(os.path.join(self.args.run_id, './images/input2/')):
                    os.mkdir(os.path.join(self.args.run_id, './images/input2/'))
                # input_save(fts[0], masks[0], targets[0], os.path.join(self.args.run_id, './images/input/'))
                # input_save(fts[1], masks[1], targets[1], os.path.join(self.args.run_id, './images/input2/'))
                # self.kspace_model.module.train_mode_set(False)
                self.kspace_model.eval()
                batch, num_frames, num_coils, numr, numc = undersampled_fts.shape
                inpt_mag_log = (og_coiled_fts.abs()+EPS).log()
                inpt_phase = og_coiled_fts / inpt_mag_log.exp()
                inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)

                tot_vids_per_patient = (dset.num_vids_per_patient*dset.frames_per_vid_per_patient)
                num_vids = 1
                num_plots = sum(tot_vids_per_patient[:num_vids]*num_coils)

                mask = None
                predr, ans_phase, ans_mag_log, loss_mag, loss_phase, loss_real, (_,_,_) = self.kspace_model(undersampled_fts[:num_vids], mask, self.device, periods[:num_vids].clone(), targ_phase = inpt_phase[:num_vids], targ_mag_log = inpt_mag_log[:num_vids], targ_real = og_coiled_vids[:num_vids])
                # predr, ans_phase, ans_mag_log, loss_mag, loss_phase, loss_real, (_,_,_) = self.kspace_model(undersampled_fts[:num_vids], mask, self.device, periods[:num_vids].clone(), targ_phase = None, targ_mag_log = None, targ_real = None)

                with torch.no_grad():
                    pred_ft = ans_phase*ans_mag_log.unsqueeze(-1).exp()
                    pred_ft = torch.complex(pred_ft[:,:,:,:,:,0],pred_ft[:,:,:,:,:,1])
                    # predr = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).real

                tot = 0
                with tqdm(total=num_plots) as pbar:
                    for bi in range(undersampled_fts[:num_vids].shape[0]):
                        for f_num in range(undersampled_fts.shape[1]):
                            for c_num in range(num_coils):
                                if num_plots == 0:
                                    return
                                num_plots -= 1
                                targi = og_coiled_vids[bi,f_num, c_num].squeeze().cpu().numpy()
                                predi = predr[bi,f_num,c_num].squeeze().cpu().numpy()
                                p_num, v_num = indices[bi]
                                orig_fti = (og_coiled_fts[bi,f_num,c_num].abs()+1).log()
                                pred_fti = (pred_ft[bi,f_num,c_num].abs()+1).log()
                                # mask_fti = (undersampled_fts[bi,f_num,c_num].abs()+1).log()
                                mask_fti = (1+undersampled_fts[bi,f_num,c_num].abs()).log()
                                
                                ifft_of_undersamp = torch.fft.ifft2(torch.fft.ifftshift(undersampled_fts[bi,f_num,c_num], dim = (-2,-1))).abs().squeeze()

                                fig = plt.figure(figsize = (16,8))
                                plt.subplot(2,4,1)
                                myimshow(orig_fti, cmap = 'gray')
                                plt.title('Original FFT')
                                plt.subplot(2,4,2)
                                myimshow((mask_fti).squeeze(), cmap = 'gray')
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
                                myimshow(ifft_of_undersamp, cmap = 'gray', trim = True)
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
                                if not os.path.exists(os.path.join(path, './patient_{}/by_location_number/location_{}/coil_{}'.format(p_num, v_num, c_num))):
                                    os.mkdir(os.path.join(path, './patient_{}/by_location_number/location_{}/coil_{}'.format(p_num, v_num, c_num)))
                                if not os.path.exists(os.path.join(path, './patient_{}/by_frame_number/'.format(p_num))):
                                    os.mkdir(os.path.join(path, './patient_{}/by_frame_number/'.format(p_num)))
                                if not os.path.exists(os.path.join(path, './patient_{}/by_frame_number/frame_{}'.format(p_num, f_num))):
                                    os.mkdir(os.path.join(path, './patient_{}/by_frame_number/frame_{}'.format(p_num, f_num)))
                                spec = ''
                                if f_num < self.parameters['init_skip_frames']:
                                    spec = 'Loss Skipped'
                                plt.suptitle("Patient {} Location {} Frame {} Coil {}\n{}".format(p_num, v_num, f_num, c_num, spec))
                                plt.savefig(os.path.join(path, './patient_{}/by_location_number/location_{}/coil_{}/frame_{}.jpg'.format(p_num, v_num, c_num, f_num)))
                                plt.savefig(os.path.join(path, './patient_{}/by_location_number/location_{}/coil_{}/frame_{}.jpg'.format(p_num, v_num, c_num, f_num)))
                                plt.savefig(os.path.join(path, './patient_{}/by_frame_number/frame_{}/location_{}_coil_{}.jpg'.format(p_num, f_num, v_num, c_num)))
                                plt.close('all')

                            tot += 1
                            pbar.update(tot)


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
            for i, (indices, undersampled_fts, og_coiled_fts, og_coiled_vids, og_video, periods) in enumerate(dloader):
                self.kspace_model.eval()
                self.ispace_model.eval()
                
                batch, num_frames, num_coils, numr, numc = undersampled_fts.shape
                
                inpt_mag_log = (og_coiled_fts.abs()+EPS).log()
                inpt_phase = og_coiled_fts / inpt_mag_log.exp()
                inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)

                tot_vids_per_patient = (dset.num_vids_per_patient*dset.frames_per_vid_per_patient)
                num_vids = 2
                batch = num_vids
                num_plots = sum(tot_vids_per_patient[:num_vids])

                mask = None
                predr, ans_phase, ans_mag_log, loss_mag, loss_phase, loss_real, (_,_,_) = self.kspace_model(undersampled_fts[:num_vids], mask, self.device, periods[:num_vids].clone(), targ_phase = None, targ_mag_log = None, targ_real = None)

                pred_ft = ans_phase*ans_mag_log.unsqueeze(-1).exp()
                pred_ft = torch.complex(pred_ft[:,:,:,:,:,0],pred_ft[:,:,:,:,:,1])

                predr[torch.isnan(predr)] = 0

                predr = predr.reshape(batch*num_frames,num_coils,numr, numc).to(self.device)
                targ_vid = og_video[:num_vids].reshape(batch*num_frames,1, numr, numc).to(self.device)
                ispace_outp = self.ispace_model(predr).cpu()

                predr = predr.reshape(batch,num_frames,num_coils,numr,numc)
                ispace_outp = ispace_outp.reshape(batch,num_frames,numr,numc)

                tot = 0
                with tqdm(total=num_plots) as pbar:
                    for bi in range(num_vids):
                        p_num, v_num = indices[bi]
                        for f_num in range(undersampled_fts.shape[1]):

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
                            og_vidi = og_video[bi, f_num,:,:]
                            ispace_outpi = ispace_outp[bi, f_num, :,:]
                            
                            plt.subplot(1,3,1)
                            myimshow(og_vidi, cmap = 'gray')
                            plt.title('Input Video')
                            # print(f_num, 1)
                            
                            plt.subplot(1,3,2)
                            myimshow(ispace_outpi, cmap = 'gray', trim = True)
                            plt.title('Ispace Prediction')
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


                            fig = plt.figure(figsize = (32,4*num_coils-2))
                            
                            plt.subplot(8,8,(((num_coils//2)-1)*8)+1)
                            myimshow(og_vidi, cmap = 'gray')
                            plt.title('Input Video')
                            # print(f_num, 1)
                            
                            plt.subplot(8,8,(((num_coils//2))*8))
                            myimshow(ispace_outpi, cmap = 'gray', trim = True)
                            plt.title('Ispace Prediction')
                            # print(f_num, 2)
                            
                            plt.subplot(8,8,(((num_coils//2)+1)*8))
                            diffvals = show_difference_image(ispace_outpi, og_vidi)
                            plt.title('Difference Frame')
                            # print(f_num, 3)

                            # plt.subplot(8,8,(((num_coils//2)+2)*8))
                            # plt.hist(diffvals, range = [0,0.25], density = False, bins = 30)
                            # plt.ylabel('Pixel Count')
                            # plt.xlabel('Difference Value')
                            # print(f_num, 4)

                            for c_num in range(num_coils):
                                if num_plots == 0:
                                    return
                                num_plots -= 1
                                
                                targi = og_coiled_vids[bi,f_num, c_num].squeeze().cpu().numpy()
                                orig_fti = (og_coiled_fts[bi,f_num,c_num].abs()+1).log()
                                mask_fti = (1+undersampled_fts[bi,f_num,c_num].abs()).log()
                                ifft_of_undersamp = torch.fft.ifft2(torch.fft.ifftshift(undersampled_fts[bi,f_num,c_num], dim = (-2,-1))).abs().squeeze()
                                pred_fti = (pred_ft[bi,f_num,c_num].abs()+1).log()
                                predi = predr[bi,f_num,c_num].squeeze().cpu().numpy()

                                plt.subplot(8,8,8*c_num+2)
                                myimshow(targi, cmap = 'gray')
                                if c_num == 0:
                                    plt.title('Coiled Input')
                                # print(c_num,1)
                                
                                plt.subplot(8,8,8*c_num+3)
                                myimshow((orig_fti).squeeze(), cmap = 'gray')
                                if c_num == 0:
                                    plt.title('FT of Coiled Input')
                                # print(c_num,2)
                                
                                plt.subplot(8,8,8*c_num+4)
                                myimshow(mask_fti, cmap = 'gray')
                                if c_num == 0:
                                    plt.title('Undersampled FT')
                                # print(c_num,3)
                                
                                plt.subplot(8,8,8*c_num+5)
                                myimshow(ifft_of_undersamp, cmap = 'gray')
                                if c_num == 0:
                                    plt.title('IFFT of Undersampled FT')
                                # print(c_num,4)
                                
                                plt.subplot(8,8,8*c_num+6)
                                myimshow(pred_fti, cmap = 'gray')
                                if c_num == 0:
                                    plt.title('FT predicted by Kspace Model')
                                # print(c_num,5)
                                
                                plt.subplot(8,8,8*c_num+7)
                                myimshow(predi, cmap = 'gray',trim = True)
                                if c_num == 0:
                                    plt.title('IFFT of Kspace Prediction')
                                # print(c_num,6)
                                
                            spec = ''
                            if f_num < self.parameters['init_skip_frames']:
                                spec = 'Loss Skipped'
                            plt.suptitle("Epoch {}\nPatient {} Video {} Frame {}\n{}".format(epoch,p_num, v_num, f_num, spec))
                            plt.savefig(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}.jpg'.format(p_num, v_num, f_num)))
                            plt.savefig(os.path.join(path, './patient_{}/by_frame_number/frame_{}/location_{}.jpg'.format(p_num, f_num, v_num)))
                            plt.close('all')

                            tot += 1
                            pbar.update(tot)