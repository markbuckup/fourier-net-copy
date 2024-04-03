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
from utils.models.periodLSTM import gaussian_2d, mylog

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

def special_trim(x, l = 5, u = 95):
    percentile_95 = np.percentile(x.detach().cpu(), u)
    percentile_5 = np.percentile(x.detach().cpu(), l)
    x = x.clip(percentile_5, percentile_95)
    return x

def torch_trim(x):
    B,C,row,col = x.shape
    
    with torch.no_grad():
        x = x.reshape(B,C,row*col)
        x = x - x.min(2, keepdim = True)[0]
        x = x / (x.max(2, keepdim = True)[0] + 1e-10)

        percentile_95 = torch.quantile(x, .95, dim = 2, keepdim = True)
        percentile_5 = torch.quantile(x, .05, dim = 2, keepdim = True)

        x = x - percentile_5
        x[x<0] = 0
        x = x + percentile_5

        x = x - percentile_95
        x[x>0] = 0
        x = x + percentile_95

        x = x - x.min(2, keepdim = True)[0]
        x = x / (x.max(2, keepdim = True)[0] + 1e-10)

        return x.reshape(B,C,row,col)

def show_difference_image(im1, im2):
    # im1 = (im1 - im1.min())
    # im1 = (im1 / (im1.max() + EPS))
    # im2 = (im2 - im2.min())
    # im2 = (im2 / (im2.max() + EPS))
    diff = (im1-im2)
    plt.axis('off')
    plt.imshow(np.abs(diff), cmap = 'plasma', vmin=0, vmax=0.25)
    # plt.colorbar()
    return np.abs(diff).reshape(-1)

class Trainer(nn.Module):
    def __init__(self, kspace_model, ispace_model, ispace_trainset, ispace_testset, trainset, testset, parameters, device, ddp_rank, ddp_world_size, args):
        super(Trainer, self).__init__()
        self.kspace_model = kspace_model
        self.ispace_model = ispace_model
        self.ispace_trainset = ispace_trainset
        self.ispace_testset = ispace_testset
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
        if not args.eval:
            self.ispace_train_sampler = DistributedSampler(
                                    self.ispace_trainset, 
                                    num_replicas=self.ddp_world_size, 
                                    rank=self.ddp_rank, 
                                    shuffle=False, 
                                    drop_last=False
                                )
            self.ispace_test_sampler = DistributedSampler(
                                    self.ispace_testset, 
                                    num_replicas=self.ddp_world_size, 
                                    rank=self.ddp_rank, 
                                    shuffle=False, 
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
        if not args.eval:
            self.ispacetrainloader = torch.utils.data.DataLoader(
                                self.ispace_trainset,
                                batch_size=self.parameters['train_batch_size'],
                                shuffle = False,
                                num_workers = 0,
                                pin_memory = False,
                                drop_last = False,
                                sampler = self.ispace_train_sampler
                            )
            self.ispacetestloader = torch.utils.data.DataLoader(
                                self.ispace_testset,
                                batch_size=self.parameters['test_batch_size'],
                                shuffle = False,
                                num_workers = 0,
                                pin_memory = False,
                                drop_last = False,
                                sampler = self.ispace_test_sampler
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
                    self.kspace_optim = optim.Adam(list(self.kspace_model.module.kspace_m.parameters())+list(self.kspace_model.module.ispacem.parameters()), lr=self.parameters['lr_kspace'], betas=self.parameters['optimizer_params'])
                else:
                    self.kspace_optim = optim.Adam(self.kspace_model.module.kspace_m.parameters(), lr=self.parameters['lr_kspace'], betas=self.parameters['optimizer_params'])
            self.ispace_optim = optim.Adam(self.ispace_model.parameters(), lr=self.parameters['lr_ispace'], betas=self.parameters['optimizer_params'])
            self.parameters['scheduler_params']['cycle_momentum'] = False
            
        if self.parameters['scheduler'] == 'None':
            self.kspace_scheduler = None
            self.ispace_scheduler = None
        if self.parameters['scheduler'] == 'StepLR':
            mydic = self.parameters['scheduler_params']
            if self.ddp_rank == 0:
                if self.parameters['kspace_architecture'] == 'KLSTM1':
                    self.kspace_scheduler = optim.lr_scheduler.StepLR(self.kspace_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=mydic['verbose'])
                self.ispace_scheduler = optim.lr_scheduler.StepLR(self.ispace_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=mydic['verbose'])
            else:
                if self.parameters['kspace_architecture'] == 'KLSTM1':
                    self.kspace_scheduler = optim.lr_scheduler.StepLR(self.kspace_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=False)
                self.ispace_scheduler = optim.lr_scheduler.StepLR(self.ispace_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=False)
        if self.parameters['scheduler'] == 'CyclicLR':
            mydic = self.parameters['scheduler_params']
            ispace_mydic = self.parameters['ispace_scheduler_params']
            if self.ddp_rank == 0:
                if self.parameters['kspace_architecture'] == 'KLSTM1':
                    self.kspace_scheduler = optim.lr_scheduler.CyclicLR(self.kspace_optim, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'],  verbose=False)
                self.ispace_scheduler = optim.lr_scheduler.CyclicLR(self.ispace_optim, ispace_mydic['base_lr'], ispace_mydic['max_lr'], step_size_up=ispace_mydic['step_size_up'], mode=ispace_mydic['mode'], cycle_momentum = ispace_mydic['cycle_momentum'],  verbose=False)
            else:
                if self.parameters['kspace_architecture'] == 'KLSTM1':
                    self.kspace_scheduler = optim.lr_scheduler.CyclicLR(self.kspace_optim, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'])
                self.ispace_scheduler = optim.lr_scheduler.CyclicLR(self.ispace_optim, ispace_mydic['base_lr'], ispace_mydic['max_lr'], step_size_up=ispace_mydic['step_size_up'], mode=ispace_mydic['mode'], cycle_momentum = ispace_mydic['cycle_momentum'])

        self.l1loss = fetch_loss_function('L1',self.device, self.parameters['loss_params'])
        # self.l2loss = fetch_loss_function('L2',self.device, self.parameters['loss_params'])
        self.SSIM = kornia.metrics.SSIM(11)
        self.msssim_loss = kornia.losses.SSIMLoss(11).to(device)
        # self.msssim_loss = kornia.losses.MS_SSIMLoss().to(device)
        # if self.criterion_FT is not None:
        #     self.criterion_FT = self.criterion_FT.to(self.device)
        # if self.criterion_reconFT is not None:
        #     self.criterion_reconFT = self.criterion_reconFT.to(self.device)

    def train(self, epoch, print_loss = False):
        if epoch < self.parameters['num_epochs_kspace']:
            self.kspace_mode = True
        else:
            self.kspace_mode = False
        if epoch >= (self.parameters['num_epochs_total'] - self.parameters['num_epochs_ispace']):
            self.ispace_mode = True
        else:
            self.ispace_mode = False
        avgkspacelossphase = 0.
        avgkspacelossreal = 0.
        avgkspacelossforget_gate = 0.
        avgkspacelossinput_gate = 0.
        avgkspacelossmag = 0.
        kspacessim_score = 0.
        avgkspace_l1_loss = 0.
        avgkspace_l2_loss = 0.
        avgispacelossreal = 0.
        sosssim_score = 0.
        avgsos_l1_loss = 0.
        avgsos_l2_loss = 0.
        ispacessim_score = 0.
        avgispace_l1_loss = 0.
        avgispace_l2_loss = 0.
        if self.ispace_mode and not self.args.eval:
            self.ispacetrainloader.sampler.set_epoch(epoch)
            dset = self.ispace_trainset
            dloader = self.ispacetrainloader
        else:
            self.trainloader.sampler.set_epoch(epoch)
            dset = self.trainset
            dloader = self.trainloader
        if self.ddp_rank == 0:
            if self.ispace_mode and self.kspace_mode:
                tqdm_object = tqdm(enumerate(self.trainloader), total = len(self.trainloader), desc = "[{}] | KS+IS Epoch {}/{}".format(os.getpid(), epoch+1, self.parameters['num_epochs_ispace']), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
            elif self.ispace_mode:
                tqdm_object = tqdm(enumerate(self.ispacetrainloader), total = len(self.ispacetrainloader), desc = "[{}] | IS Epoch {}/{}".format(os.getpid(), epoch+1, self.parameters['num_epochs_ispace']), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
                # tqdm_object = tqdm(enumerate(self.ispace_trainloader), total = len(self.trainloader), desc = "[{}] | IS Epoch {}/{}".format(os.getpid(), epoch+1, self.parameters['num_epochs_ispace']), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
            else:
                tqdm_object = tqdm(enumerate(self.trainloader), total = len(self.trainloader), desc = "[{}] | KS Epoch {}/{}".format(os.getpid(), epoch+1, self.parameters['num_epochs_kspace']), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
        else:
            if self.ispace_mode:
                tqdm_object = enumerate(self.ispacetrainloader)
            else:
                tqdm_object = enumerate(self.trainloader)
        for i, data_instance in tqdm_object:
            if self.kspace_mode:
                (indices, masks, og_video, undersampled_fts, coils_used, periods) = data_instance
                skip_kspace = False
            else:
                mem = data_instance[0]
                if mem == 0:
                    skip_kspace = False
                    # print('no skip!')
                    (indices, masks, og_video, undersampled_fts, coils_used, periods) = data_instance[1:]
                else:
                    skip_kspace = True
                    # print('skip!')
                    predr, targ_vid = data_instance[1:]
                    predr = predr.to(self.device)
                    targ_vid = targ_vid.to(self.device)

            if not skip_kspace:
                with torch.set_grad_enabled(self.kspace_mode):
                    if self.kspace_mode:
                        self.kspace_optim.zero_grad(set_to_none=True)

                    batch, num_frames, chan, numr, numc = undersampled_fts.shape
                    if self.parameters['kspace_combine_coils']:
                        og_coiled_vids = og_video.to(self.device)
                        og_fts = torch.fft.fftshift(torch.fft.fft2(og_video.to(self.device)), dim = (-2,-1))
                        inpt_mag_log = mylog((og_fts.abs()+EPS), base = self.parameters['logarithm_base'])
                        inpt_phase = og_fts / (self.parameters['logarithm_base']**inpt_mag_log)
                        inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)
                    else:
                        og_coiled_vids = og_video.to(self.device) * coils_used.to(self.device)
                        # og_coiled_vids = og_coiled_vids - og_coiled_vids.min(-1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1)
                        # og_coiled_vids = og_coiled_vids / (og_coiled_vids.max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1) + EPS)
                        og_coiled_fts = torch.fft.fftshift(torch.fft.fft2(og_coiled_vids), dim = (-2,-1))
                        inpt_mag_log = mylog((og_coiled_fts.abs()+EPS), base = self.parameters['logarithm_base'])
                        inpt_phase = og_coiled_fts / (self.parameters['logarithm_base']**inpt_mag_log)
                        inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)
                    
                    if not (self.parameters['skip_kspace_lstm'] and (not self.parameters['ispace_lstm'])):
                        if self.kspace_mode:
                            self.kspace_model.train()
                        else:
                            self.kspace_model.eval()

                    if not (self.parameters['skip_kspace_lstm'] and (not self.parameters['ispace_lstm'])):
                        predr, _, _, loss_mag, loss_phase, loss_real, loss_forget_gate, loss_input_gate, (loss_l1, loss_l2, ss1) = self.kspace_model(undersampled_fts, masks, self.device, periods.clone(), targ_phase = inpt_phase, targ_mag_log = inpt_mag_log, targ_real = og_coiled_vids, og_video = og_video)

                        predr_sos = ((predr**2).sum(2, keepdim = True) ** 0.5)[:,self.parameters['init_skip_frames']:]
                        targ_vid = og_video.to(self.device)[:,self.parameters['init_skip_frames']:]
                        loss_ss1_sos = self.msssim_loss(predr_sos.reshape(predr_sos.shape[0]*predr_sos.shape[1]*predr_sos.shape[2],1,*predr_sos.shape[3:]), targ_vid.reshape(predr_sos.shape[0]*predr_sos.shape[1]*predr_sos.shape[2],1,*predr_sos.shape[3:]))
                         
                        del masks
                        del inpt_phase
                        del inpt_mag_log
                        if not self.parameters['end-to-end-supervision']:
                            if self.parameters['kspace_real_loss_only']:
                                loss = 10*loss_real
                            else:
                                # loss = 10*loss_real
                                # print(0.06*loss_mag , 2*loss_phase , 12*loss_real, 0.2*loss_ss1_sos)
                                loss = 0.1*loss_mag + 2*loss_phase + 12*loss_real + 0.2*loss_ss1_sos
                                if self.parameters['lstm_forget_gate_loss']:
                                    loss += loss_forget_gate * 8
                                if self.parameters['lstm_input_gate_loss']:
                                    loss += loss_input_gate * 18

                                # print(0.1*loss_mag , 2*loss_phase ,12*loss_real)
                                # print(0.06*loss_mag,100*loss_phase,5*loss_real)

                            if self.kspace_mode:
                                loss.backward()
                                self.kspace_optim.step()

                            del loss
                        avgkspacelossphase += float(loss_phase.cpu().item()/(len(dloader)))
                        avgkspacelossmag += float(loss_mag.cpu().item()/(len(dloader)))
                        avgkspacelossforget_gate += float(loss_forget_gate.cpu().item()/(len(dloader)))
                        if not self.parameters['lstm_input_proc_identity']:
                            avgkspacelossinput_gate += float(loss_input_gate.cpu().item()/(len(dloader)))
                        del loss_phase
                        del loss_mag
                    else:
                        predr = torch.fft.ifft2(torch.fft.ifftshift(undersampled_fts, dim = (-2,-1))).abs().to(self.device)
                        loss_real = (predr- og_coiled_vids).reshape(predr.shape[0]*predr.shape[1]*predr.shape[2], predr.shape[3]*predr.shape[4]).abs().detach().cpu().mean(1).sum()
                        ss1 = self.SSIM(predr.reshape(predr.shape[0]*predr.shape[1]*predr.shape[2],1,predr.shape[3], predr.shape[4]), (og_coiled_vids).reshape(predr.shape[0]*predr.shape[1]*predr.shape[2],1,predr.shape[3], predr.shape[4]))
                        ss1 = ss1.reshape(ss1.shape[0],-1)
                        ss1 = ss1.mean(1).sum().detach().cpu()

                        loss_l1 = (predr- (og_coiled_vids)).reshape(predr.shape[0]*predr.shape[1]*predr.shape[2], predr.shape[3]*predr.shape[4]).abs().detach().cpu().mean(1).sum()
                        loss_l2 = (((predr- (og_coiled_vids)).reshape(predr.shape[0]*predr.shape[1]*predr.shape[2], predr.shape[3]*predr.shape[4]) ** 2)).detach().cpu().mean(1).sum()
                    avgkspacelossreal += float(loss_real.cpu().item()/(len(dloader)))/2
                    kspacessim_score += float(ss1.cpu()/dset.total_unskipped_frames)*len(self.args.gpu)
                    avgkspace_l1_loss += float(loss_l1.cpu()/dset.total_unskipped_frames)*len(self.args.gpu)
                    avgkspace_l2_loss += float(loss_l2.cpu()/dset.total_unskipped_frames)*len(self.args.gpu)
                    del loss_real
                
                with torch.no_grad():
                    predr_sos = predr_sos.clip(0,1)
                    # predr_sos = predr_sos - predr_sos.min(-1, keepdim = True)[0].min(-2, keepdim = True)[0]
                    # predr_sos = predr_sos / (EPS + predr_sos.max(-1, keepdim = True)[0].max(-2, keepdim = True)[0])

                    loss_l1_sos = (predr_sos - targ_vid).reshape(predr_sos.shape[0]*predr_sos.shape[1], predr_sos.shape[3]*predr_sos.shape[4]).abs().mean(1).sum().detach().cpu()
                    loss_l2_sos = (((predr_sos - targ_vid).reshape(predr_sos.shape[0]*predr_sos.shape[1], predr_sos.shape[3]*predr_sos.shape[4]) ** 2).mean(1).sum()).detach().cpu()
                    # ss1_sos = self.SSIM(targ_vid.reshape(predr_sos.shape[0]*predr_sos.shape[1],1,*predr_sos.shape[3:]), targ_vid.reshape(predr_sos.shape[0]*predr_sos.shape[1],1,*predr_sos.shape[3:]))
                    ss1_sos = self.SSIM(predr_sos.reshape(predr_sos.shape[0]*predr_sos.shape[1]*predr_sos.shape[2],1,*predr_sos.shape[3:]), targ_vid.reshape(predr_sos.shape[0]*predr_sos.shape[1]*predr_sos.shape[2],1,*predr_sos.shape[3:]))
                    ss1_sos = ss1_sos.reshape(ss1_sos.shape[0],-1)
                    loss_ss1_sos = ss1_sos.mean(1).sum().detach().cpu()
                    sosssim_score += float(loss_ss1_sos.cpu().item()/dset.total_unskipped_frames)*len(self.args.gpu)*self.parameters['num_coils']
                    avgsos_l1_loss += float(loss_l1_sos.cpu().item()/dset.total_unskipped_frames)*len(self.args.gpu)*self.parameters['num_coils']
                    avgsos_l2_loss += float(loss_l2_sos.cpu().item()/dset.total_unskipped_frames)*len(self.args.gpu)*self.parameters['num_coils']

                if self.parameters['coil_combine'] == 'SOS':
                    predr = ((predr**2).sum(2, keepdim = True) ** 0.5)[:,self.parameters['init_skip_frames']:]
                    predr = predr.clip(0,1)
                    # predr = predr - predr.min(-1, keepdim = True)[0].min(-2, keepdim = True)[0]
                    # predr = predr / (EPS + predr.max(-1, keepdim = True)[0].max(-2, keepdim = True)[0])
                    chan = 1

                # print('setting indices ')
                # print(indices[:,0], indices[:,1])
                # print('\n')
                if self.ispace_mode:
                    self.ispace_trainset.bulk_set_data(indices[:,0], indices[:,1], predr, targ_vid)

            
            with torch.set_grad_enabled(self.ispace_mode):

                batch, num_frames, chan, numr, numc = predr.shape

                if self.ispace_mode:
                    self.ispace_optim.zero_grad(set_to_none=True)
                
                predr = predr.reshape(batch*num_frames,chan,numr, numc).to(self.device)
                
                targ_vid = targ_vid.reshape(batch*num_frames,1, numr, numc).to(self.device)
                
                # outp = self.ispace_model(predr.detach())
                outp = self.ispace_model(predr.detach())

                # print(predr.min(), predr.max())
                # print(outp.min(), outp.max())
                # print(targ_vid.min(), targ_vid.max())
                # print('\n')
                
                # print(predr.shape)
                # print(outp.shape)
                # print(targ_vid.shape)

                # for i in range(10):
                #     plt.imsave('{}_predr.jpg'.format(i), predr.detach().cpu()[i,0,:,:], cmap = 'gray')
                #     plt.imsave('{}_outp.jpg'.format(i), outp.detach().cpu()[i,0,:,:], cmap = 'gray')
                #     plt.imsave('{}_targ_vid.jpg'.format(i), targ_vid.detach().cpu()[i,0,:,:], cmap = 'gray')
                # asdf

                if self.parameters['crop_loss']:
                    mask = gaussian_2d((self.parameters['image_resolution'],self.parameters['image_resolution'])).reshape(1,1,self.parameters['image_resolution'],self.parameters['image_resolution'])
                else:
                    mask = np.ones((1,1,self.parameters['image_resolution'],self.parameters['image_resolution']))

                mask = torch.FloatTensor(mask).to(outp.device)


                if self.ispace_mode:
                    loss_ss1 = self.msssim_loss(outp.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]), targ_vid.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]))
                    loss = 0.2*loss_ss1
                    loss += self.l1loss(outp*mask, targ_vid*mask)

                    # ss1 = self.SSIM(outp.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]), targ_vid.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]))
                    # ss1 = ss1.reshape(ss1.shape[0],-1)
                    # loss = ss1.mean(1).sum()
                    loss.backward()
                    self.ispace_optim.step()
                    avgispacelossreal += float(loss.cpu().item()/(len(dloader)))


                if self.parameters['end-to-end-supervision']:
                    self.kspace_optim.step()

                outp = outp.clip(0,1)
                # outp = outp - outp.min(-1, keepdim = True)[0].min(-2, keepdim = True)[0]
                # outp = outp / (EPS + outp.max(-1, keepdim = True)[0].max(-2, keepdim = True)[0])

                loss_l1 = (outp- targ_vid).reshape(outp.shape[0]*outp.shape[1], outp.shape[2]*outp.shape[3]).abs().mean(1).sum().detach().cpu()
                loss_l2 = (((outp- targ_vid).reshape(outp.shape[0]*outp.shape[1], outp.shape[2]*outp.shape[3]) ** 2).mean(1).sum()).detach().cpu()
                # ss1 = self.SSIM(targ_vid.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]), targ_vid.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]))
                # ss1 = self.SSIM(outp.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]), targ_vid.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]))
                # ss1 = ss1.reshape(ss1.shape[0],-1)
                ss1 = self.SSIM(outp.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]), targ_vid.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]))
                loss_ss1 = ss1.reshape(ss1.shape[0],-1).mean(1).sum().detach().cpu()
                ispacessim_score += float(loss_ss1.cpu().item()/dset.total_unskipped_frames)*len(self.args.gpu)*self.parameters['num_coils']
                avgispace_l1_loss += float(loss_l1.cpu().item()/dset.total_unskipped_frames)*len(self.args.gpu)*self.parameters['num_coils']
                avgispace_l2_loss += float(loss_l2.cpu().item()/dset.total_unskipped_frames)*len(self.args.gpu)*self.parameters['num_coils']


            if self.parameters['scheduler'] == 'CyclicLR':
                if self.kspace_mode:
                    if not (self.parameters['skip_kspace_lstm'] and (not self.parameters['ispace_lstm'])):
                        if self.parameters['kspace_architecture'] == 'KLSTM1':
                            if self.kspace_scheduler is not None:
                                self.kspace_scheduler.step()

                if self.ispace_mode:
                    if self.ispace_scheduler is not None and (self.ispace_mode):
                        self.ispace_scheduler.step()
        
        if not self.parameters['scheduler'] == 'CyclicLR':
            if self.kspace_mode:
                if not (self.parameters['skip_kspace_lstm'] and (not self.parameters['ispace_lstm'])):
                    if self.parameters['kspace_architecture'] == 'KLSTM1':
                        if self.kspace_scheduler is not None:
                            self.kspace_scheduler.step()
            if self.ispace_mode:
                if self.ispace_scheduler is not None:
                    self.ispace_scheduler.step()

        return avgkspacelossmag, avgkspacelossphase, avgkspacelossreal,avgkspacelossforget_gate, avgkspacelossinput_gate, kspacessim_score, avgkspace_l1_loss, avgkspace_l2_loss, sosssim_score, avgsos_l1_loss, avgsos_l2_loss, avgispacelossreal, ispacessim_score, avgispace_l1_loss, avgispace_l2_loss

    def evaluate(self, epoch, train = False, print_loss = False):
        if epoch < self.parameters['num_epochs_kspace']:
            self.kspace_mode = True
        else:
            self.kspace_mode = False
        if epoch >= (self.parameters['num_epochs_total'] - self.parameters['num_epochs_ispace']):
            self.ispace_mode = True
        else:
            self.ispace_mode = False
        avgkspacelossphase = 0.
        avgkspacelossreal = 0.
        avgkspacelossforget_gate = 0
        avgkspacelossinput_gate = 0
        avgkspacelossmag = 0.
        kspacessim_score = 0.
        avgkspace_l1_loss = 0.
        avgkspace_l2_loss = 0.
        avgispacelossreal = 0.
        sosssim_score = 0.
        avgsos_l1_loss = 0.
        avgsos_l2_loss = 0.
        ispacessim_score = 0.
        avgispace_l1_loss = 0.
        avgispace_l2_loss = 0.
        if train:
            self.traintestloader.sampler.set_epoch(epoch)
            dloader = self.traintestloader
            dset = self.trainset
            dstr = 'Train'
        else:
            dstr = 'Test'
            if self.ispace_mode and (not self.args.eval):
                self.ispacetestloader.sampler.set_epoch(epoch)
                dloader = self.ispacetestloader
                dset = self.ispace_testset
            else:
                self.testloader.sampler.set_epoch(epoch)
                dloader = self.testloader
                dset = self.testset
        
        if self.ddp_rank == 0:
            tqdm_object = tqdm(enumerate(dloader), total = len(dloader), desc = "Testing after Epoch {} on {}set".format(epoch, dstr), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
        else:
            tqdm_object = enumerate(dloader)
        with torch.no_grad():

            for i, data_instance in tqdm_object:
                if self.kspace_mode or self.args.eval:
                    (indices, masks, og_video, undersampled_fts, coils_used, periods) = data_instance
                    skip_kspace = False
                else:
                    mem = data_instance[0]
                    if mem == 0:
                        skip_kspace = False
                        # print('no skip!')
                        (indices, masks, og_video, undersampled_fts, coils_used, periods) = data_instance[1:]
                    else:
                        skip_kspace = True
                        # print('skip!')
                        predr, targ_vid = data_instance[1:]
                        predr = predr.to(self.device)
                        targ_vid = targ_vid.to(self.device)

                if not skip_kspace:
                    batch, num_frames, chan, numr, numc = undersampled_fts.shape
                    if self.parameters['kspace_combine_coils']:
                        og_coiled_vids = og_video.to(self.device)
                        og_fts = torch.fft.fftshift(torch.fft.fft2(og_video.to(self.device)), dim = (-2,-1))
                        inpt_mag_log = mylog((og_fts.abs()+EPS), base = self.parameters['logarithm_base'])
                        inpt_phase = og_fts / (self.parameters['logarithm_base']**inpt_mag_log)
                        inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)
                    else:
                        og_coiled_vids = og_video.to(self.device) * coils_used.to(self.device)
                        # og_coiled_vids = og_coiled_vids - og_coiled_vids.min(-1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1)
                        # og_coiled_vids = og_coiled_vids / (og_coiled_vids.max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1) + EPS)
                        og_coiled_fts = torch.fft.fftshift(torch.fft.fft2(og_coiled_vids), dim = (-2,-1))
                        inpt_mag_log = mylog((og_coiled_fts.abs()+EPS), base = self.parameters['logarithm_base'])
                        inpt_phase = og_coiled_fts / (self.parameters['logarithm_base']**inpt_mag_log)
                        inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)

                    
                    if not (self.parameters['skip_kspace_lstm'] and (not self.parameters['ispace_lstm'])):
                        self.kspace_model.eval()
                        predr, _, _, loss_mag, loss_phase, loss_real, loss_forget_gate, loss_input_gate, (loss_l1, loss_l2, ss1) = self.kspace_model(undersampled_fts, masks, self.device, periods.clone(), targ_phase = inpt_phase, targ_mag_log = inpt_mag_log, targ_real = og_coiled_vids, og_video = og_video)
                        avgkspacelossphase += float(loss_phase.item()/(len(dloader)))
                        avgkspacelossmag += float(loss_mag.item()/(len(dloader)))
                        avgkspacelossforget_gate += float(loss_forget_gate.item()/(len(dloader)))
                        if not self.parameters['lstm_input_proc_identity']:
                            avgkspacelossinput_gate += float(loss_input_gate.item()/(len(dloader)))
                    else:
                        predr = torch.fft.ifft2(torch.fft.ifftshift(undersampled_fts, dim = (-2,-1))).abs().to(self.device)
                        loss_real = (predr- (og_coiled_vids)).reshape(predr.shape[0]*predr.shape[1]*predr.shape[2], predr.shape[3]*predr.shape[4]).abs().detach().cpu().mean(1).sum()
                        ss1 = self.SSIM(predr.reshape(predr.shape[0]*predr.shape[1]*predr.shape[2],1,predr.shape[3], predr.shape[4]), (og_coiled_vids).reshape(predr.shape[0]*predr.shape[1]*predr.shape[2],1,predr.shape[3], predr.shape[4]))
                        ss1 = ss1.reshape(ss1.shape[0],-1)
                        ss1 = ss1.mean(1).sum().detach().cpu()

                        loss_l1 = (predr- (og_coiled_vids)).reshape(predr.shape[0]*predr.shape[1]*predr.shape[2], predr.shape[3]*predr.shape[4]).abs().detach().cpu().mean(1).sum()
                        loss_l2 = (((predr- (og_coiled_vids)).reshape(predr.shape[0]*predr.shape[1]*predr.shape[2], predr.shape[3]*predr.shape[4]) ** 2)).detach().cpu().mean(1).sum()


                    avgkspacelossreal += float(loss_real.item()/(len(dloader)))

                    
                    kspacessim_score += float(ss1.cpu()/dset.total_unskipped_frames)*len(self.args.gpu)
                    avgkspace_l1_loss += float(loss_l1.cpu()/dset.total_unskipped_frames)*len(self.args.gpu)
                    avgkspace_l2_loss += float(loss_l2.cpu()/dset.total_unskipped_frames)*len(self.args.gpu)

                    predr_sos = ((predr**2).sum(2, keepdim = True) ** 0.5)[:,self.parameters['init_skip_frames']:]
                    predr_sos = predr_sos.clip(0,1)
                    # predr_sos = predr_sos - predr_sos.min(-1, keepdim = True)[0].min(-2, keepdim = True)[0]
                    # predr_sos = predr_sos / (EPS + predr_sos.max(-1, keepdim = True)[0].max(-2, keepdim = True)[0])

                    targ_vid = og_video.to(self.device)[:,self.parameters['init_skip_frames']:]
                    
                    loss_l1_sos = (predr_sos - targ_vid).reshape(predr_sos.shape[0]*predr_sos.shape[1], predr_sos.shape[3]*predr_sos.shape[4]).abs().mean(1).sum().detach().cpu()
                    loss_l2_sos = (((predr_sos - targ_vid).reshape(predr_sos.shape[0]*predr_sos.shape[1], predr_sos.shape[3]*predr_sos.shape[4]) ** 2).mean(1).sum()).detach().cpu()
                    ss1_sos = self.SSIM(predr_sos.reshape(predr_sos.shape[0]*predr_sos.shape[1],1,*predr_sos.shape[3:]), targ_vid.reshape(predr_sos.shape[0]*predr_sos.shape[1],1,*predr_sos.shape[3:]))
                    # ss1_sos = self.SSIM(targ_vid.reshape(targ_vid.shape[0]*predr_sos.shape[1],1,*predr_sos.shape[3:]), targ_vid.reshape(predr_sos.shape[0]*predr_sos.shape[1],1,*predr_sos.shape[3:]))
                    ss1_sos = ss1_sos.reshape(ss1_sos.shape[0],-1)
                    loss_ss1_sos = ss1_sos.mean(1).sum().detach().cpu()
                    sosssim_score += float(loss_ss1_sos.cpu().item()/(dset.total_unskipped_frames/self.parameters['num_coils']))*len(self.args.gpu)
                    avgsos_l1_loss += float(loss_l1_sos.cpu().item()/(dset.total_unskipped_frames/self.parameters['num_coils']))*len(self.args.gpu)
                    avgsos_l2_loss += float(loss_l2_sos.cpu().item()/(dset.total_unskipped_frames/self.parameters['num_coils']))*len(self.args.gpu)

                    if self.parameters['coil_combine'] == 'SOS':
                        predr = predr_sos
                        predr = predr.clip(0,1)
                        # predr = predr - predr.min(-1, keepdim = True)[0].min(-2, keepdim = True)[0]
                        # predr = predr / (EPS + predr.max(-1, keepdim = True)[0].max(-2, keepdim = True)[0])
                        chan = 1
                    else:
                        predr = predr[:,self.parameters['init_skip_frames']:]

                    if self.ispace_mode and not self.args.eval:
                        self.ispace_testset.bulk_set_data(indices[:,0], indices[:,1], predr, targ_vid)

                batch, num_frames, chan, numr, numc = predr.shape

                if self.parameters['kspace_combine_coils']:
                    chan = 1

                predr = predr.reshape(batch*num_frames,chan,numr, numc).to(self.device)
                targ_vid = targ_vid.reshape(batch*num_frames,1, numr, numc).to(self.device)
                    
                self.ispace_model.eval()
                outp = self.ispace_model(predr.detach())

                loss = self.l1loss(outp, targ_vid)


                outp = outp.clip(0,1)
                # outp = outp - outp.min(-1, keepdim = True)[0].min(-2, keepdim = True)[0]
                # outp = outp / (EPS + outp.max(-1, keepdim = True)[0].max(-2, keepdim = True)[0])

                loss_l1 = ((outp)- (targ_vid)).reshape(outp.shape[0]*outp.shape[1], outp.shape[2]*outp.shape[3]).abs().detach().cpu().mean(1)
                loss_l2 = ((((outp)- (targ_vid)).reshape(outp.shape[0]*outp.shape[1], outp.shape[2]*outp.shape[3]) ** 2)).detach().cpu().mean(1)
                ss1 = self.SSIM((outp).reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]), (targ_vid).reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]))
                ss1 = ss1.reshape(ss1.shape[0],-1)
                loss_ss1 = ss1.mean(1).sum().detach().cpu()
                loss_l1 = loss_l1.sum()
                loss_l2 = loss_l2.sum()

                avgispacelossreal += float(loss.cpu().item()/(len(dloader)))*len(self.args.gpu)
                ispacessim_score += float(loss_ss1.cpu().item()/(dset.total_unskipped_frames/8))*len(self.args.gpu)
                avgispace_l1_loss += float(loss_l1.cpu().item()/(dset.total_unskipped_frames/8))*len(self.args.gpu)
                avgispace_l2_loss += float(loss_l2.cpu().item()/(dset.total_unskipped_frames/8))*len(self.args.gpu)


        return avgkspacelossmag, avgkspacelossphase, avgkspacelossreal, avgkspacelossforget_gate, avgkspacelossinput_gate, kspacessim_score, avgkspace_l1_loss, avgkspace_l2_loss, sosssim_score, avgsos_l1_loss, avgsos_l2_loss, avgispacelossreal, ispacessim_score, avgispace_l1_loss, avgispace_l2_loss

    def time_analysis(self):
        tqdm_object = tqdm(enumerate(self.testloader), total = len(self.testloader))
        times = []
        with torch.no_grad():
            for i, (indices, masks, og_video, undersampled_fts, coils_used, periods) in tqdm_object:
            # for i, (indices, undersampled_fts, masks, og_coiled_fts, og_coiled_vids, og_video, periods) in tqdm_object:
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
            for i, (indices, masks, og_video, undersampled_fts, coils_used, periods) in enumerate(dloader):
            # for i, (indices, undersampled_fts, masks, og_coiled_fts, og_coiled_vids, og_video, periods) in tqdm_object:
                batch, num_frames, chan, numr, numc = undersampled_fts.shape
                if self.parameters['kspace_combine_coils']:
                    og_coiled_vids = og_video.to(self.device) * coils_used.to(self.device)
                    og_coiled_fts = torch.fft.fftshift(torch.fft.fft2(og_coiled_vids.to(self.device)), dim = (-2,-1))
                    inpt_mag_log = mylog((og_coiled_fts.abs()+EPS), base = self.parameters['logarithm_base'])
                    inpt_phase = og_coiled_fts / (self.parameters['logarithm_base']**inpt_mag_log)
                    inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)
                else:
                    og_coiled_vids = og_video.to(self.device) * coils_used.to(self.device)
                    # og_coiled_vids = og_coiled_vids - og_coiled_vids.min(-1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1)
                    # og_coiled_vids = og_coiled_vids / (og_coiled_vids.max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1) + EPS)
                    og_coiled_fts = torch.fft.fftshift(torch.fft.fft2(og_coiled_vids), dim = (-2,-1))
                    inpt_mag_log = mylog((og_coiled_fts.abs()+EPS), base = self.parameters['logarithm_base'])
                    inpt_phase = og_coiled_fts / (self.parameters['logarithm_base']**inpt_mag_log)
                    inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)

                self.kspace_model.eval()
                self.ispace_model.eval()

                batch, num_frames, chan, numr, numc = undersampled_fts.shape
                
                tot_vids_per_patient = (dset.num_vids_per_patient*dset.frames_per_vid_per_patient)
                num_vids = 1
                batch = num_vids
                num_plots = num_vids*num_frames
                if not (self.parameters['skip_kspace_lstm'] and (not self.parameters['ispace_lstm'])):
                    # predr, _, _, loss_mag, loss_phase, loss_real,loss_forget_gate, loss_input_gate, (_,_,_) = self.kspace_model(undersampled_fts[:num_vids], masks[:num_vids], self.device, periods[:num_vids].clone(), targ_phase = inpt_phase, targ_mag_log = inpt_mag_log, targ_real = og_coiled_vids, og_video = og_video)
                    predr, _, _, loss_mag, loss_phase, loss_real, loss_forget_gate, loss_input_gate, (_,_,_) = self.kspace_model(undersampled_fts[:num_vids], masks[:num_vids], self.device, periods[:num_vids].clone(), targ_phase = None, targ_mag_log = None, targ_real = None, og_video = None)
                else:
                    predr = torch.fft.ifft2(torch.fft.ifftshift(undersampled_fts, dim = (-2,-1))).abs().to(self.device)

                if torch.isnan(predr).any():
                    print('Predr nan',torch.isnan(predr).any())
                predr[torch.isnan(predr)] = 0

                sos_output = (predr**2).sum(2, keepdim = False).cpu() ** 0.5
                sos_output = sos_output.clip(0,1)
                # sos_output = sos_output - sos_output.min(-1, keepdim = True)[0].min(-2, keepdim = True)[0]
                # sos_output = sos_output / (EPS + sos_output.max(-1, keepdim = True)[0].max(-2, keepdim = True)[0])

                if self.parameters['coil_combine'] == 'SOS':
                    ispace_input = (predr**2).sum(2, keepdim = True) ** 0.5
                    ispace_input = ispace_input.clip(0,1)
                    # ispace_input = ispace_input - ispace_input.min(-1, keepdim = True)[0].min(-2, keepdim = True)[0]
                    # ispace_input = ispace_input / (EPS + ispace_input.max(-1, keepdim = True)[0].max(-2, keepdim = True)[0])
                    chan = 1
                else:
                    ispace_input = predr

                if self.parameters['kspace_combine_coils']:
                    chan = 1
                
                ispace_input = ispace_input.reshape(batch*num_frames,chan,numr, numc).to(self.device)
                # print(predr.shape)
                # for j in range(10):
                #     for i in range(8):
                #         plt.imsave('{}_{}.jpg'.format(j,i), predr[j,i,:,:].cpu(), cmap = 'gray')
                # asdf
                targ_vid = og_video[:num_vids].reshape(batch*num_frames,1, numr, numc).to(self.device)
                
                # temp = og_coiled_vids.reshape(batch*num_frames,chan, numr, numc).to(self.device)
                # predr = predr - predr.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(2).detach()
                # predr = predr / (EPS + predr.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(2).detach())

                # og_video = og_video - og_video.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(2).detach()
                # og_video = og_video / (EPS + og_video.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(2).detach())
                # print(og_video.min())
                # print(og_video.max())
                # asdf

                # ispace_outp = self.ispace_model(temp).reshape(batch,num_frames,numr,numc).cpu()
                
                ispace_outp = self.ispace_model(ispace_input).cpu().reshape(batch,num_frames,numr,numc)
                
                ispace_outp = ispace_outp.clip(0,1)
                # ispace_outp = ispace_outp - ispace_outp.min(-1, keepdim = True)[0].min(-2, keepdim = True)[0]
                # ispace_outp = ispace_outp / (EPS + ispace_outp.max(-1, keepdim = True)[0].max(-2, keepdim = True)[0])

                # ispace_outp = ispace_outp - ispace_outp.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(2).detach()
                # ispace_outp = ispace_outp / (EPS + ispace_outp.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(2).detach())
                
                # # B, 1, 120, X, Y - B, 120, 1, X, Y
                # B,F,R,C = ispace_outp.shape
                # idffs = ((ispace_outp.unsqueeze(1) - og_video)**2)
                # for i in range(F):
                #     idffs[:,i,i,:] = 1000000000000
                # print((idffs).reshape(B,F,F,-1).mean(3).min(2))
                # lags = torch.arange(F).reshape(1,F) - idffs.reshape(B,F,F,-1).mean(3).min(2)[1]
                # print(lags)
                # print(lags.min())
                # asdf
                

                predr = predr.reshape(batch,num_frames,self.parameters['num_coils'],numr, numc).to(self.device)
                pred_ft = torch.fft.fftshift(torch.fft.fft2(predr), dim = (-2,-1))
                
                tot = 0
                with tqdm(total=num_plots) as pbar:
                    for bi in range(num_vids):
                        p_num, v_num = indices[bi]
                        for f_num in range(undersampled_fts.shape[1]):

                            os.makedirs(os.path.join(path, './patient_{}/by_location_number/location_{}'.format(p_num, v_num)), exist_ok=True)
                            os.makedirs(os.path.join(path, './patient_{}/by_frame_number/frame_{}'.format(p_num, f_num)), exist_ok=True)
                            
                            fig = plt.figure(figsize = (20,6))
                            og_vidi = og_video.cpu()[bi, f_num,0,:,:]
                            ispace_outpi = ispace_outp[bi, f_num, :,:]
                            sos_outpi = sos_output[bi, f_num, :,:]
                            


                            plt.subplot(1,5,1)
                            myimshow(og_vidi, cmap = 'gray')
                            plt.title('Ground Truth Frame')


                            plt.subplot(1,5,2)
                            myimshow(sos_outpi, cmap = 'gray')
                            plt.title('Kspace Prediction + SOS')
                            # print(f_num, 2)
                            
                            plt.subplot(1,5,3)
                            diffvals = show_difference_image(sos_outpi, og_vidi)
                            plt.title('Difference Frame SOS')

                            # print(f_num, 1)
                            # ispace_outpi = torch.from_numpy(match_histograms(ispace_outpi.numpy(), og_vidi.numpy(), channel_axis=None))
                            plt.subplot(1,5,4)
                            myimshow(ispace_outpi, cmap = 'gray')
                            plt.title('Ispace Prediction')
                            # print(f_num, 2)
                            
                            plt.subplot(1,5,5)
                            diffvals = show_difference_image(ispace_outpi, og_vidi)
                            plt.title('Difference Frame ISpace')
                            spec = ''
                            if f_num < self.parameters['init_skip_frames']:
                                spec = 'Loss Skipped'
                            plt.suptitle("Epoch {}\nPatient {} Video {} Frame {}\n{}".format(epoch,p_num, v_num, f_num, spec))
                            plt.savefig(os.path.join(path, './patient_{}/by_location_number/location_{}/io_frame_{}.jpg'.format(p_num, v_num, f_num)))
                            plt.savefig(os.path.join(path, './patient_{}/by_frame_number/frame_{}/io_location_{}.jpg'.format(p_num, f_num, v_num)))





                            # print(f_num, 3)

                            # plt.subplot(2,2,4)
                            # plt.hist(diffvals, range = [0,0.25], density = False, bins = 30, histtype='step')
                            # plt.ylabel('Pixel Count')
                            # plt.xlabel('Difference Value')
                            # print(f_num, 4)
                            if self.parameters['kspace_combine_coils']:
                                kspace_out_size = 1
                            else:
                                kspace_out_size = self.parameters['num_coils']
                            spec = ''
                            if f_num < self.parameters['init_skip_frames']:
                                spec = 'Loss Skipped'

                            if not self.args.ispace_visual_only:
                                fig = plt.figure(figsize = (22,4*kspace_out_size))
                                
                                # plt.subplot(kspace_out_size,7,(((kspace_out_size//2))*7)+1)
                                # myimshow(og_vidi, cmap = 'gray')
                                # plt.title('Input Video')
                                # print(f_num, 1)

                                
                                # plt.subplot(kspace_out_size,7,(((kspace_out_size//2)+1)*7))
                                # diffvals = show_difference_image(ispace_outpi, og_vidi.numpy())
                                # plt.title('Difference Frame')
                                # print(f_num, 3)

                                # plt.subplot(8,8,(((kspace_out_size//2)+2)*8))
                                # plt.hist(diffvals, range = [0,0.25], density = False, bins = 30)
                                # plt.ylabel('Pixel Count')
                                # plt.xlabel('Difference Value')
                                # print(f_num, 4)

                                num_plots -= 1
                                for c_num in range(kspace_out_size):
                                    if num_plots == 0:
                                        return
                                    
                                    targi = og_coiled_vids.cpu()[bi,f_num, c_num].squeeze().cpu().numpy()
                                    orig_fti = mylog((og_coiled_fts.cpu()[bi,f_num,c_num].abs()+1), base = self.parameters['logarithm_base'])
                                    mask_fti = mylog((1+undersampled_fts.cpu()[bi,f_num,c_num].abs()), base = self.parameters['logarithm_base'])
                                    ifft_of_undersamp = torch.fft.ifft2(torch.fft.ifftshift(undersampled_fts.cpu()[bi,f_num,c_num], dim = (-2,-1))).abs().squeeze()
                                    pred_fti = mylog((pred_ft.cpu()[bi,f_num,c_num].abs()+1), base = self.parameters['logarithm_base'])
                                    predi = predr.cpu()[bi,f_num,c_num].squeeze().cpu().numpy()

                                    plt.subplot(kspace_out_size,7,7*c_num+1)
                                    myimshow(targi, cmap = 'gray')
                                    if c_num == 0:
                                        plt.title('Coiled Input')
                                    # print(c_num,1)
                                    
                                    plt.subplot(kspace_out_size,7,7*c_num+2)
                                    myimshow((orig_fti).squeeze(), cmap = 'gray')
                                    if c_num == 0:
                                        plt.title('FT of Coiled Input')
                                    # print(c_num,2)
                                    
                                    plt.subplot(kspace_out_size,7,7*c_num+3)
                                    myimshow(mask_fti, cmap = 'gray')
                                    if c_num == 0:
                                        plt.title('Undersampled FT')
                                    # print(c_num,3)
                                    
                                    # plt.subplot(kspace_out_size,7,7*c_num+5)
                                    # myimshow(ifft_of_undersamp, cmap = 'gray')
                                    # if c_num == 0:
                                    #     plt.title('IFFT of Undersampled FT')
                                    # # print(c_num,4)
                                    
                                    plt.subplot(kspace_out_size,7,7*c_num+4)
                                    myimshow(pred_fti, cmap = 'gray')
                                    if c_num == 0:
                                        plt.title('FT predicted by Kspace Model')
                                    # print(c_num,5)
                                    
                                    plt.subplot(kspace_out_size,7,7*c_num+5)
                                    myimshow(predi, cmap = 'gray')
                                    if c_num == 0:
                                        plt.title('IFFT of Kspace Prediction')
                                    # print(c_num,6)
                                    
                                    plt.subplot(kspace_out_size,7,7*c_num+6)
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
