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

from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms, models, datasets

import sys
sys.path.append('/root/Cardiac-MRI-Reconstrucion/')

from utils.functions import fetch_loss_function


class Trainer(nn.Module):
    def __init__(self, model, trainset, testset, parameters, device):
        super(Trainer, self).__init__()
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.parameters = parameters
        self.device = device
        self.trainloader = torch.utils.data.DataLoader(
                            trainset, 
                            batch_size=self.parameters['train_batch_size'], 
                            shuffle = True
                        )
        self.traintestloader = torch.utils.data.DataLoader(
                            trainset, 
                            batch_size=self.parameters['train_batch_size'], 
                            shuffle = False
                        )
        self.testloader = torch.utils.data.DataLoader(
                            testset, 
                            batch_size=self.parameters['test_batch_size'],
                            shuffle = False)

        if self.parameters['optimizer'] == 'Adam':
            self.optim = optim.Adam(self.model.parameters(), lr=self.parameters['lr'], betas=self.parameters['optimizer_params'])
        elif self.parameters['optimizer'] == 'SGD':
            mom, wt_dec = self.parameters['optimizer_params']
            self.optim = optim.SGD(self.model.parameters(), lr=self.parameters['lr'], momentum = mom, weight_decay = wt_dec)
            
        if self.parameters['scheduler'] == 'None':
            self.scheduler = None
        if self.parameters['scheduler'] == 'StepLR':
            mydic = self.parameters['scheduler_params']
            self.scheduler = optim.lr_scheduler.StepLR(self.optim, mydic['step_size'], gamma=mydic['gamma'], verbose=mydic['verbose'])

        self.criterion = fetch_loss_function(self.parameters['loss_recon'], self.device, self.parameters['loss_params']).to(self.device)
        self.criterion_FT = fetch_loss_function(self.parameters['loss_FT'], self.device, self.parameters['loss_params'])
        self.criterion_reconFT = fetch_loss_function(self.parameters['loss_reconstructed_FT'], self.device, self.parameters['loss_params'])

    def train(self, epoch):
        avglossrecon = 0
        avglossft = 0
        avglossreconft = 0
        beta1 = self.parameters['beta1']
        beta2 = self.parameters['beta2']
        for i, (indices, fts, fts_masked, targets, target_fts) in tqdm(enumerate(self.trainloader), total = len(self.trainloader), desc = "[{}] | Epoch {}".format(os.getpid(), epoch)):
            self.optim.zero_grad()
            # print('Computing forward', fts_masked.shape, flush = True)
            # t = time.process_time()
            ft_preds, preds = self.model(fts_masked.to(self.device)) # B, 1, X, Y
            # print('Computed forward', time.process_time()-t, flush = True)
            # print('Computing loss', flush = True)
            # t = time.process_time()
            loss_recon = self.criterion(preds.to(self.device), targets.to(self.device))
            # print('Computed loss', time.process_time()-t, flush = True)
            loss_ft = torch.tensor([0]).to(self.device)
            loss_reconft = torch.tensor([0]).to(self.device)
            if self.criterion_FT is not None:
                loss_ft = self.criterion_FT(ft_preds.to(self.device), target_fts.to(self.device))
            if self.criterion_reconFT is not None:
                if self.parameters['loss_reconstructed_FT'] == 'Cosine-Watson':
                    loss_reconft = self.criterion_reconFT(preds.to(self.device), targets.to(self.device))
                else:
                    predfft = torch.fft.fft2(preds.to(self.device)).log()
                    predfft = torch.stack((predfft.real, predfft.imag),-1)
                    targetfft = torch.fft.fft2(targets.to(self.device)).log()
                    targetfft = torch.stack((targetfft.real, targetfft.imag),-1)
                    loss_reconft = self.criterion_reconFT(predfft, targetfft)
            loss = loss_recon + beta1*loss_ft + beta2*loss_reconft
            # print('Computing backward', fts_masked.shape, flush = True)
            # t = time.process_time()      
            loss.backward()
            self.optim.step()
            # print('Computed backward', time.process_time()-t, flush = True)
            avglossrecon += loss_recon.item()/(len(self.trainloader))
            avglossft += loss_ft.item()/(len(self.trainloader))
            avglossreconft += loss_reconft.item()/(len(self.trainloader))

        print('Average Recon Loss for Epoch {} = {}' .format(epoch, avglossrecon), flush = True)
        if self.criterion_FT is not None:
            print('Average FT Loss for Epoch {} = {}' .format(epoch, avglossft), flush = True)
        if self.criterion_reconFT is not None:
            print('Average Recon FT Loss for Epoch {} = {}' .format(epoch, avglossreconft), flush = True)
        return avglossrecon, avglossft, avglossreconft

    def evaluate(self, epoch, train = False):
        if train:
            dloader = self.traintestloader
            dset = self.trainset
            dstr = 'Train'
        else:
            dloader = self.testloader
            dset = self.testset
            dstr = 'Test'
        avglossrecon = 0
        avglossft = 0
        avglossreconft = 0
        with torch.no_grad():
            for i, (indices, fts, fts_masked, targets, target_fts) in tqdm(enumerate(dloader), total = len(dloader), desc = "Testing after Epoch {} on {}set".format(epoch, dstr)):
                ft_preds, preds = self.model(fts_masked.to(self.device))
                avglossrecon += self.criterion(preds.to(self.device), targets.to(self.device)).item()/(len(dloader))
                if self.criterion_FT is not None:
                    avglossft += self.criterion_FT(ft_preds.to(self.device), target_fts.to(self.device)).item()/(len(dloader))
                if self.criterion_reconFT is not None:
                    if self.parameters['loss_reconstructed_FT'] == 'Cosine-Watson':
                        avglossreconft += self.criterion_reconFT(preds.to(self.device), targets.to(self.device)).item()/(len(dloader))
                    else:
                        predfft = torch.fft.fft2(preds.to(self.device)).log()
                        predfft = torch.stack((predfft.real, predfft.imag),-1)
                        targetfft = torch.fft.fft2(targets.to(self.device)).log()
                        targetfft = torch.stack((targetfft.real, targetfft.imag),-1)
                        avglossreconft += self.criterion_reconFT(predfft, targetfft).item()/(len(dloader))

        if self.scheduler is not None:
            self.scheduler.step()
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
            path = './results/train'
        else:
            dloader = self.testloader
            dset = self.testset
            dstr = 'Test'
            path = './results/test'
        print('Saving plots for {} data'.format(dstr), flush = True)
        with torch.no_grad():
            for i, (indices, fts, fts_masked, targets, target_fts) in enumerate(dloader):
                ft_preds, preds = self.model(fts_masked.to(self.device))
                break
            for i in range(num_plots):
                targi = targets[i].squeeze().cpu().numpy()
                predi = preds[i].squeeze().cpu().numpy()
                # fig = plt.figure(figsize = (8,8))
                # plt.subplot(2,2,1)
                # ft = torch.complex(fts_masked[i,0,3,:,:,0],fts_masked[i,0,3,:,:,1])
                # plt.imshow(ft.abs(), cmap = 'gray')
                # plt.title('Undersampled FFT Frame')
                # plt.subplot(2,2,2)
                # plt.imshow(torch.fft.ifft2(torch.fft.ifftshift(ft.exp())).real, cmap = 'gray')
                # plt.title('IFFT of the Input')
                # plt.subplot(2,2,3)
                # plt.imshow(predi, cmap = 'gray')
                # plt.title('Our Predicted Frame')
                # plt.subplot(2,2,4)
                # plt.imshow(targi, cmap = 'gray')
                # plt.title('Actual Frame')
                # plt.suptitle("{} data window index {}".format(dstr, indices[i]))
                # plt.savefig(os.path.join(path, '{}_result_epoch{}_{}'.format(dstr, epoch, i)))
                # plt.close('all')
                fig = plt.figure(figsize = (8,4))
                # plt.subplot(2,2,1)
                # ft = torch.complex(fts_masked[i,0,3,:,:,0],fts_masked[i,0,3,:,:,1])
                # plt.imshow(ft.abs(), cmap = 'gray')
                # plt.title('Undersampled FFT Frame')
                # plt.subplot(2,2,2)
                # plt.imshow(torch.fft.ifft2(torch.fft.ifftshift(ft.exp())).real, cmap = 'gray')
                # plt.title('IFFT of the Input')
                plt.subplot(1,2,1)
                plt.imshow(predi, cmap = 'gray')
                plt.title('Our Predicted Frame')
                plt.subplot(1,2,2)
                plt.imshow(targi, cmap = 'gray')
                plt.title('Actual Frame')
                plt.suptitle("{} data window index {}".format(dstr, indices[i]))
                plt.savefig(os.path.join(path, '{}_result_epoch{}_{}'.format(dstr, epoch, i)))
                plt.close('all')