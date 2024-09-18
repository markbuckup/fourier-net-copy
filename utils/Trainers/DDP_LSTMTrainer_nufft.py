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
from torchvision import transforms, models, datasets
from torch.utils.data.distributed import DistributedSampler

EPS = 1e-10
CEPS = torch.complex(torch.tensor(EPS),torch.tensor(EPS)).exp()

import sys
sys.path.append('../../')

from utils.functions import fetch_loss_function
from utils.models.periodLSTM import gaussian_2d, mylog

def myimshow(x, cmap = 'gray', trim = False):
    """
    Displays an image with optional trimming.

    Parameters:
    ------------
    - x : np.ndarray
        Image array to be displayed.
    - cmap : str
        Colormap used for display. 
            Default is 'gray'.
    - trim : bool
        Whether to trim the image intensities at 5th and 95th percentiles. 
            Default is False.
    """
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
    """
    Trims the values of a tensor to be within the specified percentiles.

    Parameters:
    ------------
    - x : torch.Tensor
        Tensor to be trimmed.
    - l : int
        Lower percentile for trimming. 
            Default is 5.
    - u : int
        Upper percentile for trimming. 
            Default is 95.

    Returns:
    ----------
    - torch.Tensor
        Trimmed tensor.
    ======================
    """
    percentile_95 = np.percentile(x.detach().cpu(), u)
    percentile_5 = np.percentile(x.detach().cpu(), l)
    x = x.clip(percentile_5, percentile_95)
    return x

def torch_trim(x):
    """
    Normalizes and trims a 4D tensor along its last two dimensions.

    Parameters:
    ------------
    - x : torch.Tensor
        Input tensor of shape (B, C, H, W).

    Returns:
    ----------
    - torch.Tensor
        Normalized and trimmed tensor.
    """
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
    """
    Displays the absolute difference between two images.

    Parameters:
    ------------
    - im1 : np.ndarray
        First image array.
    - im2 : np.ndarray
        Second image array.

    Returns:
    ----------
    - np.ndarray
        Absolute difference image reshaped to a 1D array.
    """
    diff = (im1-im2)
    plt.axis('off')
    plt.imshow(np.abs(diff), cmap = 'plasma', vmin=0, vmax=0.5)
    plt.colorbar()
    return np.abs(diff).reshape(-1)

class Trainer(nn.Module):
    """
    Trainer class for handling the training and evaluation of models in a distributed setup.

    Attributes:
    ------------
    - recurrent_model : nn.Module
        The recurrent module consisting of the k-space RNN and the image lstm.
    - coil_combine_unet : nn.Module
        The U-Net model for combining coils.
    - trainset : Dataset
        Training dataset.
    - testset : Dataset
        Testing dataset.
    - ispace_trainset : Dataset
        Training dataset that can memoise the recurrent module prediction for a fast U-Net training.
    - ispace_testset : Dataset
        Testing dataset that can memoise the recurrent module prediction for a fast U-Net training
    - parameters : dict
        Dictionary of training parameters loaded from params.py
    - device : torch.device
        Device on which the training is conducted.
    - ddp_rank : int 
        Rank of the current process in distributed training.
    - ddp_world_size : int 
        Total number of processes in distributed training.
    - args : Namespace
        argparse arguments
    """

    def __init__(self, recurrent_model, coil_combine_unet, ispace_trainset, ispace_testset, trainset, testset, parameters, device, ddp_rank, ddp_world_size, args):
        super(Trainer, self).__init__()
        self.recurrent_model = recurrent_model
        self.coil_combine_unet = coil_combine_unet
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
        if not args.eval_on_real:
            self.train_sampler = DistributedSampler(
                                    self.trainset, 
                                    num_replicas=self.ddp_world_size, 
                                    rank=self.ddp_rank, 
                                    shuffle=True, 
                                    drop_last=False
                                )
        if not args.eval and (not ispace_trainset is None) and not self.args.eval_on_real:
            self.ispace_train_sampler = DistributedSampler(
                                    self.ispace_trainset, 
                                    num_replicas=self.ddp_world_size, 
                                    rank=self.ddp_rank, 
                                    shuffle=True, 
                                    drop_last=False
                                )
            self.ispace_test_sampler = DistributedSampler(
                                    self.ispace_testset, 
                                    num_replicas=self.ddp_world_size, 
                                    rank=self.ddp_rank, 
                                    shuffle=False, 
                                    drop_last=False
                                )
        if not args.eval_on_real:
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
        if not args.eval and (not ispace_trainset is None) and not self.args.eval_on_real:
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
        if not args.eval_on_real:
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
            if self.parameters['image_lstm']:       # AERS: The optimizer has access to the weights and we pass it the model. It will then calcualte: weight = weight - learning_rate*weight.gradient
                self.recurrent_optim = optim.Adam(list(self.recurrent_model.module.kspace_m.parameters())+list(self.recurrent_model.module.ispacem.parameters()), lr=self.parameters['lr_kspace'], betas=(0.9, 0.999))
            else:
                self.recurrent_optim = optim.Adam(self.recurrent_model.module.kspace_m.parameters(), lr=self.parameters['lr_kspace'], betas=(0.9, 0.999))
            self.unet_optim = optim.Adam(self.coil_combine_unet.parameters(), lr=self.parameters['lr_ispace'], betas=(0.9, 0.999))
            self.parameters['scheduler_params']['cycle_momentum'] = False
            
        if self.parameters['scheduler'] == 'None':
            self.recurrent_scheduler = None
            self.unet_scheduler = None
        if self.parameters['scheduler'] == 'StepLR':
            mydic = self.parameters['scheduler_params']
            if self.ddp_rank == 0:
                self.recurrent_scheduler = optim.lr_scheduler.StepLR(self.recurrent_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=mydic['verbose'])
                self.unet_scheduler = optim.lr_scheduler.StepLR(self.unet_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=mydic['verbose'])
            else:
                self.recurrent_scheduler = optim.lr_scheduler.StepLR(self.recurrent_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=False)
                self.unet_scheduler = optim.lr_scheduler.StepLR(self.unet_optim, mydic['step_size'], gamma=mydic['gamma'], verbose=False)
        if self.parameters['scheduler'] == 'CyclicLR':
            mydic = self.parameters['scheduler_params']
            ispace_mydic = self.parameters['unet_scheduler_params']
            if self.ddp_rank == 0:
                self.recurrent_scheduler = optim.lr_scheduler.CyclicLR(self.recurrent_optim, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'],  verbose=False)
                self.unet_scheduler = optim.lr_scheduler.CyclicLR(self.unet_optim, ispace_mydic['base_lr'], ispace_mydic['max_lr'], step_size_up=ispace_mydic['step_size_up'], mode=ispace_mydic['mode'], cycle_momentum = ispace_mydic['cycle_momentum'],  verbose=False)
            else:
                self.recurrent_scheduler = optim.lr_scheduler.CyclicLR(self.recurrent_optim, mydic['base_lr'], mydic['max_lr'], step_size_up=mydic['step_size_up'], mode=mydic['mode'], cycle_momentum = mydic['cycle_momentum'])
                self.unet_scheduler = optim.lr_scheduler.CyclicLR(self.unet_optim, ispace_mydic['base_lr'], ispace_mydic['max_lr'], step_size_up=ispace_mydic['step_size_up'], mode=ispace_mydic['mode'], cycle_momentum = ispace_mydic['cycle_momentum'])

        self.l1loss = fetch_loss_function('L1',self.device, self.parameters['loss_params'])
        self.l2loss = fetch_loss_function('L2',self.device, self.parameters['loss_params'])
        self.SSIM = kornia.metrics.SSIM(11)
        self.msssim_loss = kornia.losses.SSIMLoss(11).to(device)

    def train(self, epoch, print_loss = False):  # AERS: Trains the epoch it is training for. print_loss is always False, so if it is True, something is wrong!
        """
        Trains the model for one epoch.

        Parameters:
        ------------
        - epoch : int
            The current epoch number.
        - print_loss : bool
            Whether to print the loss values. 
                Default is False.

        Returns:
        ---------- 
        - tuple
            Averaged losses and scores for the epoch.
        """

        # AERS: Determines if it is in k-space or image space mode: what model is training?
        if epoch < self.parameters['num_epochs_recurrent']:
            self.recurrent_mode = True
        else:
            self.recurrent_mode = False
        if epoch >= (self.parameters['num_epochs_total'] - self.parameters['num_epochs_unet']):
            self.unet_mode = True
        else:
            self.unet_mode = False

        # AERS: Initializations for the accumulators (aka the statistics that will be returner)
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

        # AERS: Select the proper dataset and dataloader, depending on the current epoch. Memoized vs normal one.
        if self.unet_mode and not self.args.eval and not self.args.eval_on_real and not self.recurrent_mode:
            self.ispacetrainloader.sampler.set_epoch(epoch)
            dset = self.ispace_trainset
            dloader = self.ispacetrainloader
            kspace_skip_frames_loss = self.ispace_trainset.parameters['init_skip_frames']
        else:
            self.trainloader.sampler.set_epoch(epoch)
            dset = self.trainset
            dloader = self.trainloader
            kspace_skip_frames_loss = self.parameters['init_skip_frames']
        if self.ddp_rank == 0:          # AERS: Only prints progress bar if we are in rank 0, to avoid printing multiple progress bars.
            if self.unet_mode and self.recurrent_mode:
                tqdm_object = tqdm(enumerate(self.trainloader), total = len(self.trainloader), desc = "[{}] | KS+IS Epoch {}/{}".format(os.getpid(), epoch+1, self.parameters['num_epochs_total']), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
            elif self.unet_mode:
                tqdm_object = tqdm(enumerate(self.ispacetrainloader), total = len(self.ispacetrainloader), desc = "[{}] | IS Epoch {}/{}".format(os.getpid(), epoch+1, self.parameters['num_epochs_total']), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
            else:
                tqdm_object = tqdm(enumerate(self.trainloader), total = len(self.trainloader), desc = "[{}] | KS Epoch {}/{}".format(os.getpid(), epoch+1, self.parameters['num_epochs_total']), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
        else:
            if self.unet_mode and not self.recurrent_mode:
                tqdm_object = enumerate(self.ispacetrainloader)
            else:
                tqdm_object = enumerate(self.trainloader)

        # AERS: tqdm object is iterating over the data
        for i, data_instance in tqdm_object: # AERS: Fetch a batch of data from the tqdm object (progress bar), which is a train loader
            if self.recurrent_mode:
                # AERS: Output from ACDC_radial_faster.py (getitem)– data instance for the k-space mode
                # AERS: indices of patient numbers, masks, gound truth video, input to the neural network, coils and period
                (indices, masks, og_video, coilwise_targets, undersampled_fts, coils_used, periods) = data_instance
                skip_kspace = False   # AERS: If you are in k-space mode, don't skip it.
            else:  # AERS: If, in image mode, then a different dataset object is used. The memoized dataset object.
                mem = data_instance[0]  # AERS: From ACDC_radial_faster.py (getitem), the first output is a boolean to indicate if data are memoized or not
                if mem == 0:            # AERS: If not memoized, data needs to be expanded into multiple variables (as above)
                    skip_kspace = False
                    (indices, masks, og_video, coilwise_targets, undersampled_fts, coils_used, periods) = data_instance[1:]
                else:
                    skip_kspace = True   # AERS: If it is not zero, skip k-space training and used the memoized data
                    predr, targ_vid = data_instance[1:]
                    predr = predr.to(self.device)
                    targ_vid = targ_vid.to(self.device)

#################### AERS: Recurrent Module Training ####################

            if not skip_kspace:          
                with torch.set_grad_enabled(self.recurrent_mode):           # AERS: Enclosure to enable or disable gradient computation for anything inside of this.
                    if self.recurrent_mode:
                        self.recurrent_optim.zero_grad(set_to_none=True)    # AERS: Default (June 10th video, min 8:03). Sets everything to None. It is faster than writing zeros.

                    # AERS: Additional preprocessing. NRM mentions shows a previous code version where there is an if to set parameters to combine coils or not. 
                    # Ground truth for k-space RNN is the coil-wise gt. Coil combination is performed by U-Net.
                    # This pre-processing is done here because the data needs to be pushed to GPU to perform FFT (faster than CPU).
                    # ACDC_radial_faster avoids uting GPU inside the dataset object to prevent GPU memory leak. 
                    # As per NRM: A dataset object should not have access to GPU at all. It should just run on the CPU.
                    # AERS: FFT of coil-wise ground truth is needed because in the k-space RNN, the loss is a loss element in both k-space and image space.
                    batch, num_frames, chan, numr, numc = undersampled_fts.shape
                    og_coiled_vids = coilwise_targets.to(self.device)                                           # AERS: coil-wise ground truth for k-space RNN
                    og_coiled_fts = torch.fft.fftshift(torch.fft.fft2(og_coiled_vids), dim = (-2,-1))           # AERS: coil-wise ground truth FFT  
                    
                    # AERS: log of the magnitude to makes sure values are in a reasonable range rather than very large. 
                    # This ensures the NN will learn with a reasonable learning rate. 
                    inpt_mag_log = mylog((og_coiled_fts.abs()+EPS), base = self.parameters['logarithm_base'])   
                    inpt_phase = og_coiled_fts / (self.parameters['logarithm_base']**inpt_mag_log)              # AERS: phase estimation
                    inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)
                    
                    # AERS: If we want to skip the k-space LSTM or not
                    if not (self.parameters['skip_kspace_rnn'] and (not self.parameters['image_lstm'])):
                        if self.recurrent_mode:
                            self.recurrent_model.train()        # AERS: Train
                        else:
                            self.recurrent_model.eval()         # AERS: Testing

                    # AERS: Generate the predictions
                    if not (self.parameters['skip_kspace_rnn'] and (not self.parameters['image_lstm'])):
                        # AERS: First forward pass or the .train()
                        # AERS: Sends undersampled input, masks, GPU #, period of each video (for ARKS), ground truths (phase and log of mag), \
                        # and original video (coil combined from previous implementation).

                        # AERS: predicted image space (output of image LSTM -> U-Net),  predr_kpace (output of k-space RNN),
                        # losses: loss_mag (L1 loss between magnitudes), loss_phase (L1 loss between angles), loss_real (iFFT of data and compare w/original coil gt)
                        predr, _, loss_mag, loss_phase, loss_real, loss_forget_gate, loss_input_gate, (loss_l1, loss_l2, ss1) = \
                            self.recurrent_model(undersampled_fts, masks, self.device, periods.clone(), targ_phase = inpt_phase, \
                                                 targ_mag_log = inpt_mag_log, targ_real = og_coiled_vids, og_video = og_video, epoch = epoch)

                        # AERS: Statistics
                        predr_sos = ((predr**2).sum(2, keepdim = True) ** 0.5)[:,kspace_skip_frames_loss:]
                        targ_vid = og_video.to(self.device)[:,kspace_skip_frames_loss:]
                        # AERS: SSIM of sum of squares prediction
                        loss_ss1_sos = self.msssim_loss(predr_sos.reshape(predr_sos.shape[0]*predr_sos.shape[1]*predr_sos.shape[2],1,*predr_sos.shape[3:]), targ_vid.reshape(predr_sos.shape[0]*predr_sos.shape[1]*predr_sos.shape[2],1,*predr_sos.shape[3:]))
                         
                        del masks
                        del inpt_phase
                        del inpt_mag_log
                        
                        # AERS: Before backprogating the loss, they need to be added together to get a combined loss. 
                        # Because they have different magnitudes, they get scaled. 
                        loss = 0.1*loss_mag + 2*loss_phase + 12*loss_real + 0.2*loss_ss1_sos
                        if self.parameters['lstm_forget_gate_loss']:
                            loss += loss_forget_gate * 8
                        if self.parameters['lstm_input_gate_loss']:
                            loss += loss_input_gate * 18

                        # AERS: Backpropagate the loss
                        if self.recurrent_mode:
                            loss.backward()                   # AERS: Propagates the loss to the leaves (weights of NN that require gradients). 
                                                              # Each number in the predictions is a function of a particular weight in the NN. 
                                                              # This will compute the gradient of the output with respect to a particular weight (dloss/dweight) and store it. 
                            self.recurrent_optim.step()       # AERS: Optimizer has access to all the weights and we pass it the model, so it will iterate all the weights to update them based on the learning rate and gradient

                        del loss

                        # AERS: Accumulator variables for losses (in CPU, to avoid using GPU memory).
                        avgkspacelossphase += float(loss_phase.cpu().item()/(len(dloader)))        # AERS: Divides by the data loader length to average across data
                        avgkspacelossmag += float(loss_mag.cpu().item()/(len(dloader)))
                        avgkspacelossforget_gate += float(loss_forget_gate.cpu().item()/(len(dloader)))
                        avgkspacelossinput_gate += float(loss_input_gate.cpu().item()/(len(dloader)))
                        del loss_phase
                        del loss_mag

                    else: # AERS: If we are skipping k-space training, the prediction should be the iFFT of the undersample data FFTs.
                          # AERS: This is bascially another experiment in which the k-space model is deleted (ablation study).
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
                
                with torch.no_grad():                   # AERS: Ensures that gradient computation regardless of any condition.
                    predr_sos = predr_sos.clip(0,1)

                    # AERS: Computes statistics before U-Net
                    loss_l1_sos = (predr_sos - targ_vid).reshape(predr_sos.shape[0]*predr_sos.shape[1], predr_sos.shape[3]*predr_sos.shape[4]).abs().mean(1).sum().detach().cpu()
                    loss_l2_sos = (((predr_sos - targ_vid).reshape(predr_sos.shape[0]*predr_sos.shape[1], predr_sos.shape[3]*predr_sos.shape[4]) ** 2).mean(1).sum()).detach().cpu()
                    ss1_sos = self.SSIM(predr_sos.reshape(predr_sos.shape[0]*predr_sos.shape[1]*predr_sos.shape[2],1,*predr_sos.shape[3:]), targ_vid.reshape(predr_sos.shape[0]*predr_sos.shape[1]*predr_sos.shape[2],1,*predr_sos.shape[3:]))
                    ss1_sos = ss1_sos.reshape(ss1_sos.shape[0],-1)
                    loss_ss1_sos = ss1_sos.mean(1).sum().detach().cpu()
                    sosssim_score += float(loss_ss1_sos.cpu().item()/dset.total_unskipped_frames)*len(self.args.gpu)*self.parameters['num_coils']
                    avgsos_l1_loss += float(loss_l1_sos.cpu().item()/dset.total_unskipped_frames)*len(self.args.gpu)*self.parameters['num_coils']
                    avgsos_l2_loss += float(loss_l2_sos.cpu().item()/dset.total_unskipped_frames)*len(self.args.gpu)*self.parameters['num_coils']

                # AERS: Coil combination. Two ways: SOS or U-Net
                if self.parameters['coil_combine'] == 'SOS':
                    predr = ((predr**2).sum(2, keepdim = True) ** 0.5)[:,kspace_skip_frames_loss:]
                    predr = predr.clip(0,1)
                    chan = 1
                else:
                    predr = predr[:,kspace_skip_frames_loss:]    # AERS: U-Net

                # ARES: If you want to memoize the image space, we bulk set the data (batches defined with indices[:,0] and indices[:,1])
                # AERS: Calls the classmethod bulk_set_data in ACDC_radial_faster.py, which has afor loop to sort the dataset into batches.
                # AERS: If not in k-space mode, and we reach here, it means that we are in image space mode and we came here to memoize the image data
                if self.unet_mode and self.parameters['memoise_ispace'] and not self.args.eval and not self.args.eval_on_real and not self.recurrent_mode:
                    self.ispace_trainset.bulk_set_data(indices[:,0], indices[:,1], predr, targ_vid)   

#################### AERS: U-Net (Image Space) Training ####################
            
            with torch.set_grad_enabled(self.unet_mode):

                batch, num_frames, chan, numr, numc = predr.shape

                if self.unet_mode:
                    self.unet_optim.zero_grad(set_to_none=True)
                
                # AERS: Prediction. Reshapes the data to the original shape (channels as number of coils)
                predr = predr.reshape(batch*num_frames,chan,numr, numc).to(self.device)
                
                # AERS: Ground truth (single coil channel)
                targ_vid = targ_vid.reshape(batch*num_frames,1, numr, numc).to(self.device)
                
                # AERS: Coil combination using U-Net
                outp = self.coil_combine_unet(predr.detach())

                # AERS: In a previous version this used to be called crop_loss and NRM said not to use it because it creates 'bright corners' in k-space. 
                # In the image domain, there was a DC shift in the log domain. This loss (crop_loss) was not useful. 
                if self.parameters['center_weighted_loss']:
                    mask = gaussian_2d((self.parameters['image_resolution'],self.parameters['image_resolution'])).reshape(1,1,self.parameters['image_resolution'],self.parameters['image_resolution'])
                else:
                    mask = np.ones((1,1,self.parameters['image_resolution'],self.parameters['image_resolution']))

                mask = torch.FloatTensor(mask).to(outp.device)

                # AERS: This loss is a combination of MS-SSIM and L1 losses. 
                if self.unet_mode:
                    loss_ss1 = self.msssim_loss(outp.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]), targ_vid.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]))
                    loss = 0.2*loss_ss1
                    loss += self.l2loss(outp*mask, targ_vid*mask)

                    loss.backward()                 # AERS: Backpropagation
                    self.unet_optim.step()          # AERS: Update the weights
                    avgispacelossreal += float(loss.cpu().item()/(len(dloader)))    


                outp = outp.clip(0,1)

                # AERS: Statistics
                loss_l1 = (outp- targ_vid).reshape(outp.shape[0]*outp.shape[1], outp.shape[2]*outp.shape[3]).abs().mean(1).sum().detach().cpu()
                loss_l2 = (((outp- targ_vid).reshape(outp.shape[0]*outp.shape[1], outp.shape[2]*outp.shape[3]) ** 2).mean(1).sum()).detach().cpu()
                ss1 = self.SSIM(outp.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]), targ_vid.reshape(outp.shape[0]*outp.shape[1],1,*outp.shape[2:]))
                loss_ss1 = ss1.reshape(ss1.shape[0],-1).mean(1).sum().detach().cpu()
                ispacessim_score += float(loss_ss1.cpu().item()/dset.total_unskipped_frames)*len(self.args.gpu)*self.parameters['num_coils']
                avgispace_l1_loss += float(loss_l1.cpu().item()/dset.total_unskipped_frames)*len(self.args.gpu)*self.parameters['num_coils']
                avgispace_l2_loss += float(loss_l2.cpu().item()/dset.total_unskipped_frames)*len(self.args.gpu)*self.parameters['num_coils']

            # AERS: If scheduler is CyclicLR, a scheduler step is performed every batch. As opposed to other schedulers where it is done every epoch.
            if self.parameters['scheduler'] == 'CyclicLR':
                if self.recurrent_mode:
                    if not (self.parameters['skip_kspace_rnn'] and (not self.parameters['image_lstm'])):
                        if self.recurrent_scheduler is not None:
                            self.recurrent_scheduler.step()

                if self.unet_mode:
                    if self.unet_scheduler is not None and (self.unet_mode):
                        self.unet_scheduler.step()
        
        # AERS: A step LR is performed for each epoch (outside the loop above).
        if not self.parameters['scheduler'] == 'CyclicLR':
            if self.recurrent_mode:
                if not (self.parameters['skip_kspace_rnn'] and (not self.parameters['image_lstm'])):
                    if self.recurrent_scheduler is not None:
                        self.recurrent_scheduler.step()
            if self.unet_mode:
                if self.unet_scheduler is not None:
                    self.unet_scheduler.step()

        # AERS: It returns a total of 15 statistics from the epoch: average k-space loss for magnitude, phase, real, forget gate, input gate, SSIM, L1, L2, etc.
        # AERS: This is only for a single GPU. These needs to be combined in DDP_paradigms_LSTM_nufft.py
        return avgkspacelossmag, avgkspacelossphase, avgkspacelossreal,avgkspacelossforget_gate, avgkspacelossinput_gate, kspacessim_score, \
            avgkspace_l1_loss, avgkspace_l2_loss, sosssim_score, avgsos_l1_loss, avgsos_l2_loss, avgispacelossreal, ispacessim_score, avgispace_l1_loss, avgispace_l2_loss

#################### AERS: Testing ####################
# This code is basically the same as the training code, without:
#  loss.backward(), optimizer.step(), optimizer.zero_grad(), 
#  and enclosed in no_grad().

    def evaluate(self, epoch, train = False, print_loss = False):
        """
        Evaluates the model after training for one epoch.

        Parameters:
        ------------
        - epoch : int
            The current epoch number.
        - train : bool
            Whether to evaluate on the training set. 
                Default is False.
        - print_loss : bool
            Whether to print the loss values. 
                Default is False.

        Returns:
        ----------
        - tuple
            Averaged losses and scores for the evaluation.
        """
        if epoch < self.parameters['num_epochs_recurrent']:
            self.recurrent_mode = True
        else:
            self.recurrent_mode = False
        if epoch >= (self.parameters['num_epochs_total'] - self.parameters['num_epochs_unet']):
            self.unet_mode = True
        else:
            self.unet_mode = False
            
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
            kspace_skip_frames_loss = self.parameters['init_skip_frames']
        else:
            dstr = 'Test'
            if self.unet_mode and (not self.args.eval) and (not self.args.eval_on_real) and not self.recurrent_mode:
                self.ispacetestloader.sampler.set_epoch(epoch)
                dloader = self.ispacetestloader
                dset = self.ispace_testset
                kspace_skip_frames_loss = self.ispace_trainset.parameters['init_skip_frames']
            else:
                self.testloader.sampler.set_epoch(epoch)
                dloader = self.testloader
                dset = self.testset
                kspace_skip_frames_loss = self.parameters['init_skip_frames']
        
        if self.ddp_rank == 0:
            tqdm_object = tqdm(enumerate(dloader), total = len(dloader), desc = "Testing after Epoch {} on {}set".format(epoch, dstr), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}")
        else:
            tqdm_object = enumerate(dloader)
        with torch.no_grad():

            for i, data_instance in tqdm_object:
                if self.recurrent_mode or self.args.eval or self.args.eval_on_real:
                    (indices, masks, og_video, coilwise_targets, undersampled_fts, coils_used, periods) = data_instance
                    skip_kspace = False
                else:
                    mem = data_instance[0]
                    if mem == 0:
                        skip_kspace = False
                        (indices, masks, og_video, coilwise_targets, undersampled_fts, coils_used, periods) = data_instance[1:]
                    else:
                        skip_kspace = True
                        predr, targ_vid = data_instance[1:]
                        predr = predr.to(self.device)
                        targ_vid = targ_vid.to(self.device)

                if not skip_kspace:
                    batch, num_frames, chan, numr, numc = undersampled_fts.shape
                    og_coiled_vids = coilwise_targets.to(self.device)
                    og_coiled_fts = torch.fft.fftshift(torch.fft.fft2(og_coiled_vids), dim = (-2,-1))
                    inpt_mag_log = mylog((og_coiled_fts.abs()+EPS), base = self.parameters['logarithm_base'])
                    inpt_phase = og_coiled_fts / (self.parameters['logarithm_base']**inpt_mag_log)
                    inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)

                    
                    if not (self.parameters['skip_kspace_rnn'] and (not self.parameters['image_lstm'])):
                        if not self.parameters['kspace_architecture'] == 'MDCNN':
                            self.recurrent_model.eval()
                        predr, _, loss_mag, loss_phase, loss_real, loss_forget_gate, loss_input_gate, (loss_l1, loss_l2, ss1) = self.recurrent_model(undersampled_fts, masks, self.device, periods.clone(), targ_phase = inpt_phase, targ_mag_log = inpt_mag_log, targ_real = og_coiled_vids, og_video = og_video)
                        avgkspacelossphase += float(loss_phase.item()/(len(dloader)))
                        avgkspacelossmag += float(loss_mag.item()/(len(dloader)))
                        avgkspacelossforget_gate += float(loss_forget_gate.item()/(len(dloader)))
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

                    predr_sos = ((predr**2).sum(2, keepdim = True) ** 0.5)[:,kspace_skip_frames_loss:]
                    predr_sos = predr_sos.clip(0,1)

                    targ_vid = og_video.to(self.device)[:,kspace_skip_frames_loss:]
                    
                    loss_l1_sos = (predr_sos - targ_vid).reshape(predr_sos.shape[0]*predr_sos.shape[1], predr_sos.shape[3]*predr_sos.shape[4]).abs().mean(1).sum().detach().cpu()
                    loss_l2_sos = (((predr_sos - targ_vid).reshape(predr_sos.shape[0]*predr_sos.shape[1], predr_sos.shape[3]*predr_sos.shape[4]) ** 2).mean(1).sum()).detach().cpu()
                    ss1_sos = self.SSIM(predr_sos.reshape(predr_sos.shape[0]*predr_sos.shape[1],1,*predr_sos.shape[3:]), targ_vid.reshape(predr_sos.shape[0]*predr_sos.shape[1],1,*predr_sos.shape[3:]))
                    ss1_sos = ss1_sos.reshape(ss1_sos.shape[0],-1)
                    loss_ss1_sos = ss1_sos.mean(1).sum().detach().cpu()
                    sosssim_score += float(loss_ss1_sos.cpu().item()/(dset.total_unskipped_frames/self.parameters['num_coils']))*len(self.args.gpu)
                    avgsos_l1_loss += float(loss_l1_sos.cpu().item()/(dset.total_unskipped_frames/self.parameters['num_coils']))*len(self.args.gpu)
                    avgsos_l2_loss += float(loss_l2_sos.cpu().item()/(dset.total_unskipped_frames/self.parameters['num_coils']))*len(self.args.gpu)

                    if self.parameters['coil_combine'] == 'SOS':
                        predr = predr_sos
                        predr = predr.clip(0,1)
                        chan = 1
                    else:
                        predr = predr[:,kspace_skip_frames_loss:]

                    if self.unet_mode and (not self.args.eval) and (not self.args.eval_on_real) and self.parameters['memoise_ispace'] and not self.recurrent_mode:
                        self.ispace_testset.bulk_set_data(indices[:,0], indices[:,1], predr, targ_vid)

                batch, num_frames, chan, numr, numc = predr.shape

                predr = predr.reshape(batch*num_frames,chan,numr, numc).to(self.device)
                targ_vid = targ_vid.reshape(batch*num_frames,1, numr, numc).to(self.device)
                
                if not self.parameters['kspace_architecture'] == 'MDCNN':
                    self.coil_combine_unet.eval()
                outp = self.coil_combine_unet(predr.detach())

                loss = self.l1loss(outp, targ_vid)


                outp = outp.clip(0,1)
                
                
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
        """
        Analyzes and prints the average time per frame for the model inference.

        Returns:
        ----------
        None
        """
        tqdm_object = tqdm(enumerate(self.testloader), total = len(self.testloader))
        times = []
        with torch.no_grad():
            for i, (indices, masks, og_video, coilwise_targets, undersampled_fts, coils_used, periods) in tqdm_object:
                self.recurrent_model.eval()
                times += self.recurrent_model.module.time_analysis(undersampled_fts, self.device, periods[0:1].clone(), self.coil_combine_unet)
        print('Average Time Per Frame = {} +- {}'.format(np.mean(times), np.std(times)), flush = True)
        scipy.io.savemat(os.path.join(self.save_path, 'fps.mat'), {'times': times})
        return


    def visualise_on_real(self,epoch):
        """
        Visualizes the model predictions on actual data and saves the plots.

        Parameters:
        ------------
        - epoch : int
            The current epoch number.

        Returns:
        ----------
        - None
        """
        print('Saving plots for actual data', flush = True)
        path = os.path.join(self.save_path, 'images/actual_data')
        os.makedirs(path, exist_ok=True)
        with torch.no_grad():
            actual_data_path = self.args.actual_data_path
            actual_data = torch.load(actual_data_path)['undersampled_data'].unsqueeze(0)

            batch, num_frames, chan, numr, numc = actual_data.shape

            if not (self.parameters['skip_kspace_rnn'] and (not self.parameters['image_lstm'])):
                # AERS: predr is the image space prediction, predr_kspace is the k-space RNN prediction
                predr, predr_kspace, loss_mag, loss_phase, loss_real, loss_forget_gate, loss_input_gate, (_,_,_) = self.recurrent_model(actual_data, None, self.device, None, targ_phase = None, targ_mag_log = None, targ_real = None, og_video = None)
            else:
                predr = torch.fft.ifft2(torch.fft.ifftshift(actual_data, dim = (-2,-1))).abs().to(self.device)
                predr_kspace = predr     

            if torch.isnan(predr).any():
                print('Predr nan',torch.isnan(predr).any())
            predr[torch.isnan(predr)] = 0

            # AERS: Sum of squares output
            sos_output = (predr**2).sum(2, keepdim = False).cpu() ** 0.5
            sos_output = sos_output.clip(0,1)

            if self.parameters['coil_combine'] == 'SOS':
                ispace_input = (predr**2).sum(2, keepdim = True) ** 0.5
                ispace_input = ispace_input.clip(0,1)
                chan = 1
            else:
                ispace_input = predr

            if self.parameters['kspace_architecture'] == 'MDCNN':
                chan = 1
            
            # AERS: U-Net output
            ispace_input = ispace_input.reshape(batch*num_frames,chan,numr, numc).to(self.device)
            ispace_outp = self.coil_combine_unet(ispace_input).cpu().reshape(batch,num_frames,numr,numc)
            ispace_outp = ispace_outp.clip(0,1)

            predr = predr.reshape(batch,num_frames,chan,numr, numc).to(self.device)
            pred_ft = torch.fft.fftshift(torch.fft.fft2(predr), dim = (-2,-1))
            
            # AERS: For loop to generate images
            with tqdm(total=num_frames) as pbar:
                for f_num in range(num_frames):
                    fig = plt.figure(figsize = (8,6))
                    ispace_outpi = ispace_outp[0, f_num, :,:]
                    sos_outpi = sos_output[0, f_num, :,:]

                    plt.subplot(1,2,1)
                    myimshow(sos_outpi, cmap = 'gray')
                    plt.title('Kspace Prediction + SOS')
                    
                    plt.subplot(1,2,2)
                    myimshow(ispace_outpi, cmap = 'gray')
                    plt.title('Ispace Prediction')

                    plt.suptitle("Epoch {}\nFrame {}".format(epoch, f_num))
                    plt.savefig(os.path.join(path, 'ispace_frame_{}.jpg'.format(f_num)))
                    


                    kspace_out_size = self.parameters['num_coils']
                    spec = ''
                    if f_num < self.parameters['init_skip_frames']:
                        spec = 'Loss Skipped'

                    if not self.args.unet_visual_only:
                        fig = plt.figure(figsize = (22,4*kspace_out_size))
                        
                        for c_num in range(kspace_out_size):

                            mask_fti = mylog((1+actual_data.cpu()[0,f_num,c_num].abs()), base = self.parameters['logarithm_base'])
                            ifft_of_undersamp = torch.fft.ifft2(torch.fft.ifftshift(actual_data.cpu()[0,f_num,c_num], dim = (-2,-1))).abs().squeeze()
                            pred_fti = mylog((pred_ft.cpu()[0,f_num,c_num].abs()+1), base = self.parameters['logarithm_base'])
                            predi = predr_kspace.cpu()[0,f_num,c_num].squeeze().cpu().numpy()
                            pred_ilstmi = predr.cpu()[0,f_num,c_num].squeeze().cpu().numpy()

                            plt.subplot(kspace_out_size,5,5*c_num+1)
                            myimshow(mask_fti, cmap = 'gray')
                            if c_num == 0:
                                plt.title('Undersampled FT')
                            
                            plt.subplot(kspace_out_size,5,5*c_num+2)
                            myimshow(ifft_of_undersamp, cmap = 'gray')
                            if c_num == 0:
                                plt.title('IFFT of Undersampled FT')
                            
                            plt.subplot(kspace_out_size,5,5*c_num+3)
                            myimshow(pred_fti, cmap = 'gray')
                            if c_num == 0:
                                plt.title('FT predicted by Kspace Model')
                            
                            plt.subplot(kspace_out_size,5,5*c_num+4)
                            myimshow(predi, cmap = 'gray')
                            if c_num == 0:
                                plt.title('IFFT of Kspace Prediction')

                            plt.subplot(kspace_out_size,5,5*c_num+5)
                            myimshow(pred_ilstmi, cmap = 'gray')
                            if c_num == 0:
                                plt.title('Image Space LSTM Prediction')
                            
                            
                        plt.suptitle("Epoch {}\nFrame {}".format(epoch, f_num))
                        plt.savefig(os.path.join(path, 'frame_{}.jpg'.format(f_num)))
                    plt.close('all')

                    pbar.update(1)

#################### AERS: Visualize  ####################

    def visualise(self, epoch, train = False):
        """
        Visualizes and saves the model predictions on the training or testing data.

        Parameters:
        ------------
        - epoch : int 
            The current epoch number.
        - train : bool
            Whether to visualize the training set. 
                Default is False.

        Returns:
        ----------
        - None
        """
        
        if train:
            dloader = self.traintestloader
            dset = self.trainset
            dstr = 'Train'
            if self.args.raw_visual_only:
                path = os.path.join(self.save_path, './images/raw/train')
            else:
                path = os.path.join(self.save_path, './images/train')
        else:
            dloader = self.testloader
            dset = self.testset
            dstr = 'Test'
            if self.args.raw_visual_only:
                path = os.path.join(self.save_path, './images/raw/test')
            else:
                path = os.path.join(self.save_path, './images/test')
        
        print('Saving plots for {} data'.format(dstr), flush = True)
        with torch.no_grad():
            for i, (indices, masks, og_video, coilwise_targets, undersampled_fts, coils_used, periods) in enumerate(dloader):
                batch, num_frames, chan, numr, numc = undersampled_fts.shape
                og_coiled_vids = coilwise_targets.to(self.device)
                og_coiled_fts = torch.fft.fftshift(torch.fft.fft2(og_coiled_vids), dim = (-2,-1))
                inpt_mag_log = mylog((og_coiled_fts.abs()+EPS), base = self.parameters['logarithm_base'])
                inpt_phase = og_coiled_fts / (self.parameters['logarithm_base']**inpt_mag_log)
                inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)

                if not self.parameters['kspace_architecture'] == 'MDCNN':
                    self.recurrent_model.eval()
                    self.coil_combine_unet.eval()

                batch, num_frames, chan, numr, numc = undersampled_fts.shape
                
                tot_vids_per_patient = (dset.num_vids_per_patient*dset.frames_per_vid_per_patient)
                num_vids = 1
                batch = num_vids
                num_plots = num_vids*num_frames
                if not (self.parameters['skip_kspace_rnn'] and (not self.parameters['image_lstm'])): # AERS: forward call of the model:  
                    # AERS: Returns the k-space and image space predictions
                    predr, predr_kspace, loss_mag, loss_phase, loss_real, loss_forget_gate, loss_input_gate, (_,_,_) = self.recurrent_model(undersampled_fts[:num_vids], masks[:num_vids], self.device, periods[:num_vids].clone(), targ_phase = None, targ_mag_log = None, targ_real = None, og_video = None)
                    if predr_kspace is None:
                        predr_kspace = predr
                else:
                    predr = torch.fft.ifft2(torch.fft.ifftshift(undersampled_fts, dim = (-2,-1))).abs().to(self.device)
                    predr_kspace = predr

                if torch.isnan(predr).any():
                    print('Predr nan',torch.isnan(predr).any())
                predr[torch.isnan(predr)] = 0

                sos_output = (predr**2).sum(2, keepdim = False).cpu() ** 0.5
                sos_output = sos_output.clip(0,1)

                if self.parameters['coil_combine'] == 'SOS':
                    ispace_input = (predr**2).sum(2, keepdim = True) ** 0.5
                    ispace_input = ispace_input.clip(0,1)
                    chan = 1
                else:
                    ispace_input = predr

                if self.parameters['kspace_architecture'] == 'MDCNN':
                    chan = 1
                
                ispace_input = ispace_input.reshape(batch*num_frames,chan,numr, numc).to(self.device)
                targ_vid = og_video[:num_vids].reshape(batch*num_frames,1, numr, numc).to(self.device)                
                ispace_outp = self.coil_combine_unet(ispace_input).cpu().reshape(batch,num_frames,numr,numc)
                ispace_outp = ispace_outp.clip(0,1)
                

                predr = predr.reshape(batch,num_frames,chan,numr, numc).to(self.device)
                pred_ft = torch.fft.fftshift(torch.fft.fft2(predr_kspace), dim = (-2,-1))
                
                tot = 0
                with tqdm(total=num_plots) as pbar:
                    for bi in range(num_vids):
                        p_num, v_num = indices[bi]
                        for f_num in range(undersampled_fts.shape[1]):

                            os.makedirs(os.path.join(path, './patient_{}/by_location_number/location_{}'.format(p_num, v_num)), exist_ok=True)
                            

                            if self.args.raw_visual_only:
                                og_vidi = og_video.cpu()[bi, f_num,0,:,:]
                                ispace_outpi = ispace_outp[bi, f_num, :,:]
                                sos_outpi = sos_output[bi, f_num, :,:]

                                os.makedirs(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}'.format(p_num, v_num, f_num)), exist_ok=True)
                                
                                plt.imsave(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}/ground_truth.jpg'.format(p_num, v_num, f_num)), og_vidi, cmap = 'gray')
                                plt.imsave(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}/kspace_sos.jpg'.format(p_num, v_num, f_num)), sos_outpi, cmap = 'gray')
                                plt.imsave(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}/ispace_pred.jpg'.format(p_num, v_num, f_num)), ispace_outpi, cmap = 'gray')
                            else:
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
                                
                                plt.subplot(1,5,3)
                                diffvals = show_difference_image(sos_outpi, og_vidi)
                                plt.title('Difference Frame SOS')

                                plt.subplot(1,5,4)
                                myimshow(ispace_outpi, cmap = 'gray')
                                plt.title('Ispace Prediction')
                                
                                plt.subplot(1,5,5)
                                diffvals = show_difference_image(ispace_outpi, og_vidi)
                                plt.title('Difference Frame ISpace')
                                spec = ''
                                if f_num < self.parameters['init_skip_frames']:
                                    spec = 'Loss Skipped'
                                plt.suptitle("Epoch {}\nPatient {} Video {} Frame {}\n{}".format(epoch,p_num, v_num, f_num, spec))
                                plt.savefig(os.path.join(path, './patient_{}/by_location_number/location_{}/io_frame_{}.jpg'.format(p_num, v_num, f_num)))





                            if self.parameters['kspace_architecture'] == 'MDCNN':
                                kspace_out_size = 1
                            else:
                                kspace_out_size = self.parameters['num_coils']
                            spec = ''
                            if f_num < self.parameters['init_skip_frames']:
                                spec = 'Loss Skipped'

                            if not self.args.unet_visual_only:

                                if self.args.raw_visual_only:
                                    for c_num in range(kspace_out_size):
                                        targi = og_coiled_vids.cpu()[bi,f_num, c_num].squeeze().cpu().numpy()
                                        orig_fti = mylog((og_coiled_fts.cpu()[bi,f_num,c_num].abs()+1), base = self.parameters['logarithm_base'])
                                        mask_fti = mylog((1+undersampled_fts.cpu()[bi,f_num,c_num].abs()), base = self.parameters['logarithm_base'])
                                        ifft_of_undersamp = torch.fft.ifft2(torch.fft.ifftshift(undersampled_fts.cpu()[bi,f_num,c_num], dim = (-2,-1))).abs().squeeze()
                                        pred_fti = mylog((pred_ft.cpu()[bi,f_num,c_num].abs()+1), base = self.parameters['logarithm_base'])
                                        predi = predr_kspace.cpu()[bi,f_num,c_num].squeeze().cpu().numpy()
                                        pred_ilstmi = predr.cpu()[0,f_num,c_num].squeeze().cpu().numpy()


                                        plt.imsave(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}/coiled_gt_coil_{}.jpg'.format(p_num, v_num, f_num, c_num)), targi, cmap = 'gray')
                                        plt.imsave(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}/coiled_gt_ft_coil_{}.jpg'.format(p_num, v_num, f_num, c_num)), orig_fti, cmap = 'gray')
                                        plt.imsave(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}/undersampled_ft_coil_{}.jpg'.format(p_num, v_num, f_num, c_num)), mask_fti, cmap = 'gray')
                                        plt.imsave(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}/undersampled_coil_{}.jpg'.format(p_num, v_num, f_num, c_num)), ifft_of_undersamp, cmap = 'gray')
                                        plt.imsave(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}/pred_kspace_ft_coil_{}.jpg'.format(p_num, v_num, f_num, c_num)), pred_fti, cmap = 'gray')
                                        plt.imsave(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}/pred_kspace_coil_{}.jpg'.format(p_num, v_num, f_num, c_num)), predi, cmap = 'gray')
                                        plt.imsave(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}/pred_image_lstm_coil_{}.jpg'.format(p_num, v_num, f_num, c_num)), pred_ilstmi, cmap = 'gray')

                                else:
                                    fig = plt.figure(figsize = (36,4*kspace_out_size))
                                    
                                    num_plots -= 1
                                    for c_num in range(kspace_out_size):
                                        if num_plots == 0:
                                            return
                                        
                                        targi = og_coiled_vids.cpu()[bi,f_num, c_num].squeeze().cpu().numpy()
                                        orig_fti = mylog((og_coiled_fts.cpu()[bi,f_num,c_num].abs()+EPS), base = self.parameters['logarithm_base'])
                                        mask_fti = mylog((undersampled_fts.cpu()[bi,f_num,c_num].abs()+EPS), base = self.parameters['logarithm_base'])
                                        ifft_of_undersamp = torch.fft.ifft2(torch.fft.ifftshift(undersampled_fts.cpu()[bi,f_num,c_num], dim = (-2,-1))).abs().squeeze()
                                        pred_fti = mylog((pred_ft.cpu()[bi,f_num,c_num].abs()+EPS), base = self.parameters['logarithm_base'])
                                        predi = predr_kspace.cpu()[bi,f_num,c_num].squeeze().cpu().numpy()
                                        pred_ilstmi = predr.cpu()[0,f_num,c_num].squeeze().cpu().numpy()

                                        plt.subplot(kspace_out_size,9,9*c_num+1)
                                        myimshow(targi, cmap = 'gray')
                                        if c_num == 0:
                                            plt.title('Coiled Ground Truth')
                                        
                                        plt.subplot(kspace_out_size,9,9*c_num+2)
                                        myimshow((orig_fti).squeeze(), cmap = 'gray')
                                        if c_num == 0:
                                            plt.title('FT of Coiled Ground Truth')
                                        
                                        plt.subplot(kspace_out_size,9,9*c_num+3)
                                        myimshow(mask_fti, cmap = 'gray')
                                        if c_num == 0:
                                            plt.title('Undersampled FT')
                                        
                                        plt.subplot(kspace_out_size,9,9*c_num+4)
                                        myimshow(ifft_of_undersamp, cmap = 'gray')
                                        if c_num == 0:
                                            plt.title('IFFT of Undersampled FT')
                                        
                                        plt.subplot(kspace_out_size,9,9*c_num+5)
                                        myimshow(pred_fti, cmap = 'gray')
                                        if c_num == 0:
                                            plt.title('FT predicted by Kspace Model')
                                        
                                        plt.subplot(kspace_out_size,9,9*c_num+6)
                                        myimshow(predi, cmap = 'gray')
                                        if c_num == 0:
                                            plt.title('IFFT of Kspace Prediction')
                                        
                                        plt.subplot(kspace_out_size,9,9*c_num+7)
                                        diffvals = show_difference_image(predi, targi)
                                        if c_num == 0:
                                            plt.title('Difference Image')

                                        plt.subplot(kspace_out_size,9,9*c_num+8)
                                        myimshow(pred_ilstmi, cmap = 'gray')
                                        if c_num == 0:
                                            plt.title('ISpace LSTM Prediction')
                                        
                                        plt.subplot(kspace_out_size,9,9*c_num+9)
                                        diffvals = show_difference_image(pred_ilstmi, targi)
                                        if c_num == 0:
                                            plt.title('Difference Image')
                                        
                                    spec = ''
                                    if f_num < self.parameters['init_skip_frames']:
                                        spec = 'Loss Skipped'
                                    plt.suptitle("Epoch {}\nPatient {} Video {} Frame {}\n{}".format(epoch,p_num, v_num, f_num, spec))
                                    plt.savefig(os.path.join(path, './patient_{}/by_location_number/location_{}/frame_{}.jpg'.format(p_num, v_num, f_num)))
                            plt.close('all')

                            tot += 1
                            pbar.update(1)
                break
