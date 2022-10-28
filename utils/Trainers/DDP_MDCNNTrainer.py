import os
import gc
import sys
import PIL
import time
import torch
import random
import pickle
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

import sys
sys.path.append('/root/Cardiac-MRI-Reconstrucion/')

from utils.functions import fetch_loss_function

def myimshow(x, cmap = None):
    x = x-x.min()
    x = x/x.max()
    if cmap is not None:
        plt.imshow(x, cmap = cmap)
    else:
        plt.imshow(x)

# takes FT, FT_mask
# num_coils, num_window, 256, 256, 2
def input_save(fts, fts_masked, targets, path):
    num_coils = fts.shape[0]
    num_windows = fts.shape[1]
    avg_FT = fts.mean(1)
    avg_FT = torch.complex(avg_FT[:,:,:,0], avg_FT[:,:,:,1])
    combined_ft_undersampled = avg_FT.clone()*0
    averager = torch.zeros(combined_ft_undersampled.shape)
    for wi in range(num_windows):
        x = torch.complex(fts_masked[:,wi,:,:,0], fts_masked[:,wi,:,:,1])
        combined_ft_undersampled += x
        averager[x != 0] += 1
    combined_ft_undersampled[averager != 0] /= averager[averager != 0]

    for coili in range(num_coils):
        fig = plt.figure(figsize = (28,16))
        iter = 1
        for wi in range(num_windows):
            plt.subplot(4,num_windows,iter)
            ft = torch.complex(fts[coili,wi,:,:,0],fts[coili,wi,:,:,1])
            myimshow(ft.abs(), cmap = 'gray')
            if wi == 3:
                plt.title('Complete FFT')
            iter += 1
        for wi in range(num_windows):
            plt.subplot(4,num_windows,iter)
            ft = torch.complex(fts[coili,wi,:,:,0],fts[coili,wi,:,:,1])
            outp = torch.fft.ifft2(torch.fft.ifftshift(ft.exp(), dim = (-2, -1))).real
            myimshow(outp, cmap = 'gray')
            if wi == 3:
                plt.title('Original Image')
            iter += 1
        for wi in range(num_windows):
            plt.subplot(4,num_windows,iter)
            ft = torch.complex(fts_masked[coili,wi,:,:,0],fts_masked[coili,wi,:,:,1])
            myimshow(ft.abs(), cmap = 'gray')
            if wi == 3:
                plt.title('Undersampled FFT')
            iter += 1
        for wi in range(num_windows):
            plt.subplot(4,num_windows,iter)
            ft = torch.complex(fts_masked[coili,wi,:,:,0],fts_masked[coili,wi,:,:,1])
            outp = torch.fft.ifft2(torch.fft.ifftshift(ft.exp(), dim = (-2, -1))).real
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
            self.optim = optim.Adam(self.model.parameters(), lr=self.parameters['lr'], betas=self.parameters['optimizer_params'])
            self.parameters['scheduler_params']['cycle_momentum'] = False
        elif self.parameters['optimizer'] == 'SGD':
            mom, wt_dec = self.parameters['optimizer_params']
            self.optim = optim.SGD(self.model.parameters(), lr=self.parameters['lr'], momentum = mom, weight_decay = wt_dec)
            self.parameters['scheduler_params']['cycle_momentum'] = True
            
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

        self.criterion = fetch_loss_function(self.parameters['loss_recon'], self.device, self.parameters['loss_params']).to(self.device)
        self.criterion_FT = fetch_loss_function(self.parameters['loss_FT'], self.device, self.parameters['loss_params'])
        self.criterion_reconFT = fetch_loss_function(self.parameters['loss_reconstructed_FT'], self.device, self.parameters['loss_params'])
        if self.criterion_FT is not None:
            self.criterion_FT = self.criterion_FT.to(self.device)
        if self.criterion_reconFT is not None:
            self.criterion_reconFT = self.criterion_reconFT.to(self.device)

    def train(self, epoch, print_loss = False):
        avglossrecon = 0
        avglossft = 0
        avglossreconft = 0
        beta1 = self.parameters['beta1']
        beta2 = self.parameters['beta2']
        self.trainloader.sampler.set_epoch(epoch)
        for i, (indices, fts, fts_masked, targets, target_fts) in tqdm(enumerate(self.trainloader), total = len(self.trainloader), desc = "[{}] | Epoch {}".format(os.getpid(), epoch)):
            self.optim.zero_grad(set_to_none=True)
            # with autocast(enabled = self.parameters['Automatic_Mixed_Precision'], dtype=torch.float32):
            # with autocast(enabled = self.parameters['Automatic_Mixed_Precision']):
                # self.model.module.train_mode_set(True)
            ft_preds, preds = self.model(fts_masked.to(self.device)) # B, 1, X, Y
            loss_recon = self.criterion(preds, targets.to(self.device))
            loss_ft = torch.tensor([0]).to(self.device)
            loss_reconft = torch.tensor([0]).to(self.device)
            if self.criterion_FT is not None:
                loss_ft = self.criterion_FT(ft_preds, target_fts.to(self.device))
            if self.criterion_reconFT is not None:
                if self.parameters['loss_reconstructed_FT'] == 'Cosine-Watson':
                    loss_reconft = self.criterion_reconFT(preds, targets.to(self.device))
                else:
                    predfft = torch.fft.fft2(preds).log()
                    predfft = torch.stack((predfft.real, predfft.imag),-1)
                    targetfft = torch.fft.fft2(targets.to(self.device)).log()
                    targetfft = torch.stack((targetfft.real, targetfft.imag),-1)
                    loss_reconft = self.criterion_reconFT(predfft, targetfft)
            loss = loss_recon + beta1*loss_ft + beta2*loss_reconft

            loss.backward()
            self.optim.step()
            # self.scaler.scale(loss).backward()
            # self.scaler.step(self.optim)
            # self.scaler.update()

            avglossrecon += loss_recon.item()/(len(self.trainloader))
            avglossft += loss_ft.item()/(len(self.trainloader))
            avglossreconft += loss_reconft.item()/(len(self.trainloader))

        if self.scheduler is not None:
            self.scheduler.step()
        if print_loss:
            print('Average Recon Loss for Epoch {} = {}' .format(epoch, avglossrecon), flush = True)
            if self.criterion_FT is not None:
                print('Average FT Loss for Epoch {} = {}' .format(epoch, avglossft), flush = True)
            if self.criterion_reconFT is not None:
                print('Average Recon FT Loss for Epoch {} = {}' .format(epoch, avglossreconft), flush = True)
        return avglossrecon, avglossft, avglossreconft

    def evaluate(self, epoch, train = False, print_loss = False):
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
        avglossrecon = 0
        avglossft = 0
        avglossreconft = 0
        with torch.no_grad():
            for i, (indices, fts, fts_masked, targets, target_fts) in tqdm(enumerate(dloader), total = len(dloader), desc = "Testing after Epoch {} on {}set".format(epoch, dstr)):
                # self.model.module.train_mode_set(False)
                ft_preds, preds = self.model(fts_masked.to(self.device))
                avglossrecon += self.criterion(preds, targets.to(self.device)).item()/(len(dloader))
                if self.criterion_FT is not None:
                    avglossft += self.criterion_FT(ft_preds, target_fts.to(self.device)).item()/(len(dloader))
                if self.criterion_reconFT is not None:
                    if self.parameters['loss_reconstructed_FT'] == 'Cosine-Watson':
                        avglossreconft += self.criterion_reconFT(preds, targets.to(self.device)).item()/(len(dloader))
                    else:
                        predfft = torch.fft.fft2(preds).log()
                        predfft = torch.stack((predfft.real, predfft.imag),-1)
                        targetfft = torch.fft.fft2(targets.to(self.device)).log()
                        targetfft = torch.stack((targetfft.real, targetfft.imag),-1)
                        avglossreconft += self.criterion_reconFT(predfft, targetfft).item()/(len(dloader))


        if print_loss:
            print('{} Loss After {} Epochs:'.format(dstr, epoch), flush = True)
            print('Recon Loss = {}'.format(avglossrecon), flush = True)
            if self.criterion_FT is not None:
                print('FT Loss = {}'.format(avglossft), flush = True)
            if self.criterion_reconFT is not None:
                print('Recon FT Loss = {}'.format(avglossreconft), flush = True)
        return avglossrecon, avglossft, avglossreconft

    def visualise(self, epoch, train = False):
        num_plots = min(self.parameters['test_batch_size'], 10)
        if train:
            dloader = self.traintestloader
            dset = self.trainset
            dstr = 'Train'
            path = os.path.join(self.args.run_id, './results/train')
        else:
            dloader = self.testloader
            dset = self.testset
            dstr = 'Test'
            path = os.path.join(self.args.run_id, './results/test')
        print('Saving plots for {} data'.format(dstr), flush = True)
        with torch.no_grad():
            for i, (indices, fts, fts_masked, targets, target_fts) in enumerate(dloader):
                if not os.path.exists(os.path.join(self.args.run_id, './results/input/')):
                    os.mkdir(os.path.join(self.args.run_id, './results/input/'))
                input_save(fts[0], fts_masked[0], targets[0], os.path.join(self.args.run_id, './results/input/'))
                # self.model.module.train_mode_set(False)
                ft_preds, preds = self.model(fts_masked.to(self.device))
                break
            for i in range(num_plots):
                targi = targets[i].squeeze().cpu().numpy()
                predi = preds[i].squeeze().cpu().numpy()
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
                fig = plt.figure(figsize = (8,4))
                # plt.subplot(2,2,1)
                # ft = torch.complex(fts_masked[i,0,3,:,:,0],fts_masked[i,0,3,:,:,1])
                # myimshow(ft.abs(), cmap = 'gray')
                # plt.title('Undersampled FFT Frame')
                # plt.subplot(2,2,2)
                # myimshow(torch.fft.ifft2(torch.fft.ifftshift(ft.exp())).real, cmap = 'gray')
                # plt.title('IFFT of the Input')
                plt.subplot(1,2,1)
                myimshow(predi, cmap = 'gray')
                plt.title('Our Predicted Frame')
                plt.subplot(1,2,2)
                myimshow(targi, cmap = 'gray')
                plt.title('Actual Frame')
                plt.suptitle("{} data window index {}".format(dstr, indices[i]))
                plt.savefig(os.path.join(path, '{}_result_epoch{}_{}'.format(dstr, epoch, i)))
                plt.close('all')