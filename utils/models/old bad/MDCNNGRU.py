import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms
import numpy as np
import os
import time
import sys
sys.path.append('../../')
import matplotlib.pyplot as plt
import utils.models.complexCNNs.cmplx_conv as cmplx_conv
import utils.models.complexCNNs.cmplx_dropout as cmplx_dropout
import utils.models.complexCNNs.cmplx_upsample as cmplx_upsample
import utils.models.complexCNNs.cmplx_activation as cmplx_activation
import utils.models.complexCNNs.radial_bn as radial_bn
from utils.models.gruComponents import IFFT_module, GRUImageSpaceDecoder, GRUImageSpaceEncoder, GRUKspaceModel, GRUImageSpaceUNet
from utils.models.gruGates import GRUGate_KSpace, GRUGate_ISpace, GRUGate_ISpace_combiner
from utils.functions import fetch_loss_function

EPS = 1e-10

class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """
    def __init__(self, parameters, gate_model):
        super().__init__()
        self.gate_model = gate_model
        self.reset_gate = self.gate_model()
        self.update_gate = self.gate_model()
        self.out_gate = self.gate_model()
    
    def forward(self, input, prev_state):    
        # data size is [batch, channel, 1(optional), height, width]
        stacked_input = torch.cat([input,prev_state], dim=1)
        update = self.update_gate([stacked_input], activation = 'sigmoid')[0]
        reset = self.reset_gate([stacked_input], activation = 'sigmoid')[0]
        stacked_input2 = torch.cat([input,prev_state*reset], dim=1)
        out_input = self.out_gate([stacked_input2], activation = 'tanh')[0]
        new_state = (prev_state * (1 - update)) + (out_input * update)
        return new_state

class MDCNNGRU(nn.Module):
    def __init__(self, parameters, device):
        super(MDCNNGRU, self).__init__()

        self.train_parameters = parameters
        self.n_layers = 2
        self.device = device
        self.num_coils = parameters['num_coils']
        self.image_space_real = parameters['image_space_real']
        if parameters['architecture'] == 'MDCNNGRU1':
            self.ifft_m = IFFT_module(parameters)
            self.int_model = lambda x:self.ifft_m(x)
            self.IUnet = lambda x:x
            self.gate_model = []
            self.gate_model.append(lambda :GRUGate_KSpace(parameters))
            self.gate_model.append(lambda :GRUGate_ISpace(parameters))
            self.hidden_chans = [parameters['num_coils'], 1]
            self.hidden_reals = [False, self.image_space_real]
        elif parameters['architecture'] == 'MDCNNGRU2':
            self.ifft_m = IFFT_module(parameters)
            self.int_model = lambda x:self.ifft_m(x)
            self.IUnet = GRUImageSpaceUNet(in_channels = 8, out_channels = 128, image_space_real = self.image_space_real)
            self.gate_model = []
            self.gate_model.append(lambda :GRUGate_KSpace(parameters))
            self.gate_model.append(lambda :GRUGate_ISpace_combiner(parameters))
            self.hidden_chans = [parameters['num_coils'], 8]
            self.hidden_reals = [False, self.image_space_real]
        else:
            print("Unrecognised GRU architecture mode '{}'".format(parameters['architecture']), flush = True)
            os._exit(1)

        cells = []
        for i in range(self.n_layers):
            cell = ConvGRUCell(parameters, self.gate_model[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

        self.criterion = fetch_loss_function(self.train_parameters['loss_recon'], self.device, self.train_parameters['loss_params']).to(self.device)
        self.criterion_FT = fetch_loss_function(self.train_parameters['loss_FT'], self.device, self.train_parameters['loss_params'])
        self.criterion_reconFT = fetch_loss_function(self.train_parameters['loss_reconstructed_FT'], self.device, self.train_parameters['loss_params'])

        self.l1loss = fetch_loss_function('L1',self.device, self.train_parameters['loss_params'])
        self.l2loss = fetch_loss_function('L2',self.device, self.train_parameters['loss_params'])
        self.ssimloss = fetch_loss_function('SSIM',self.device, self.train_parameters['loss_params'])

    def get_kspace_params(self):
        return list(self.cells[0].parameters())

    def get_ispace_params(self):
        ans = list(self.cells[1].parameters())
        if self.train_parameters['architecture'] == 'MDCNNGRU2':
            ans += list(self.IUnet.parameters())
        return ans

    def forward(self, x, target_batch, target_ft_batch, hidden=None):
        '''
        Input size - B, T, C, X, Y
        B = batch
        T = Video Length
        C = Num Coils
        X, Y = resolution
        '''
        B, T, C, X, Y, complex_dim = x.shape

        ans = None
        
        loss_recons = torch.zeros(x.shape[1]).to(self.device)
        loss_fts = torch.zeros(x.shape[1]).to(self.device)
        loss_reconfts = torch.zeros(x.shape[1]).to(self.device)
        score_ssims = torch.zeros(x.shape[1]).to(self.device)
        score_l1s = torch.zeros(x.shape[1]).to(self.device)
        score_l2s = torch.zeros(x.shape[1]).to(self.device)

        hiddens = hidden
        if hiddens is None:
            hiddens = []
            for i in range(len(self.hidden_chans)):
                if self.hidden_reals[i]:
                    hiddens.append(torch.zeros((B, self.hidden_chans[i],X,Y), device = x.device))
                else:
                    hiddens.append(torch.zeros((B, self.hidden_chans[i],X,Y,complex_dim), device = x.device))
        
        for ti in range(T):
            cell_kspace = self.cells[0]
            cell_ispace = self.cells[1]
            '''
            Input size = B, C, 1, X, Y
            Output sizes : 
            Mode1 = B, C, 1, X, Y
            Mode2 = B, C, 1, X, Y
            Mode3 = B, C, 1, X, Y
            Mode4 = B, channel, X, Y
            '''
            iter_outp_kspace = cell_kspace(x[:,ti,:,:,:].to(self.device), hiddens[0])
            ispace_in = self.int_model(iter_outp_kspace)
            iter_outp_ispace = cell_ispace(ispace_in, hiddens[1])
            hiddens[0] = iter_outp_kspace.clone()
            hiddens[1] = iter_outp_ispace.clone()

            outp = self.IUnet(iter_outp_ispace)
            if not self.image_space_real:
                outp = (outp**2).sum(-1)**0.5
            target = target_batch[:,ti:ti+1,:,:].to(self.device)
            target_ft = target_ft_batch[:,ti:ti+1,:,:].to(self.device)
            
            loss_recons[ti] += self.criterion(outp, target)
            if self.criterion_FT is not None:
                loss_fts[ti] += self.criterion_FT(outp, target_ft)
            if self.criterion_reconFT is not None:
                if self.train_parameters['loss_reconstructed_FT'] == 'Cosine-Watson':
                    loss_reconfts[ti] += self.criterion_reconFT(outp, target)
                else:
                    predfft = (torch.fft.fft2(EPS+outp)+EPS).log()
                    predfft = torch.stack((predfft.real, predfft.imag),-1)
                    targetfft = (torch.fft.fft2(EPS+target) + EPS).log()
                    targetfft = torch.stack((targetfft.real, targetfft.imag),-1)
                    loss_reconfts[ti] += self.criterion_reconFT(predfft, targetfft)

            score_ssims[ti] += (1-self.ssimloss(outp, target)).item()
            score_l1s[ti] += self.l1loss(outp, target).item()
            score_l2s[ti] += self.l2loss(outp, target).item()

            if ans is None:
                ans = outp.cpu()
            else:
                ans = torch.cat([ans, outp.cpu()], dim=1)
            

        return ans, [x.detach() for x in hiddens], loss_recons, loss_fts, loss_reconfts, score_ssims, score_l1s, score_l2s
        # return ans, hiddens

# parameters = {}
# parameters['image_resolution'] = 64
# if parameters['image_resolution'] == 256:
#     parameters['train_batch_size'] = 8
#     parameters['test_batch_size'] = 8
# elif parameters['image_resolution'] == 128:
#     parameters['train_batch_size'] = 23
#     parameters['test_batch_size'] = 23
# elif parameters['image_resolution'] == 64:
#     parameters['train_batch_size'] = 70
#     parameters['test_batch_size'] = 70
# parameters['lr_kspace'] = 1e-5
# parameters['lr_ispace'] = 3e-4
# parameters['init_skip_frames'] = 10
# parameters['num_epochs'] = 50
# parameters['architecture'] = 'ConvGRU1'
# parameters['dataset'] = 'acdc'
# parameters['train_test_split'] = 0.8
# parameters['normalisation'] = False
# parameters['window_size'] = 7
# parameters['loop_videos'] = -1
# if 'gru' in parameters['architecture']:
#     batch_sizes = [-1,[6,14],[6,13],[6,30],[7,34],[7,32]]
#     parameters['train_batch_size'] = batch_sizes[int(parameters['architecture'][-1])][parameters['image_space_real']]
#     parameters['test_batch_size'] = batch_sizes[int(parameters['architecture'][-1])][parameters['image_space_real']]
# else:
#     if parameters['image_resolution'] == 256:
#         parameters['train_batch_size'] = 8
#         parameters['test_batch_size'] = 8
#     elif parameters['image_resolution'] == 128:
#         parameters['train_batch_size'] = 23
#         parameters['test_batch_size'] = 23
#     elif parameters['image_resolution'] == 64:
#         parameters['train_batch_size'] = 70
#         parameters['test_batch_size'] = 70
# parameters['FT_radial_sampling'] = 2
# parameters['predicted_frame'] = 'middle'
# parameters['num_coils'] = 8
# parameters['dataloader_num_workers'] = 0
# parameters['optimizer'] = 'Adam'
# parameters['scheduler'] = 'StepLR'
# parameters['memoise_disable'] = False
# parameters['image_space_real'] = False
# parameters['optimizer_params'] = (0.9, 0.999)
# parameters['scheduler_params'] = {
#     'base_lr': 3e-4,
#     'max_lr': 1e-3,
#     'step_size_up': 10,
#     'mode': 'triangular',
#     'step_size': parameters['num_epochs']//3,
#     'gamma': 0.5,
#     'verbose': True
# }
# parameters['loss_recon'] = 'L2'
# parameters['loss_FT'] = 'None'
# parameters['loss_reconstructed_FT'] = 'None'
# parameters['beta1'] = 1
# parameters['beta2'] = 0.5
# parameters['Automatic_Mixed_Precision'] = False
# parameters['loss_params'] = {
#     'SSIM_window': 11,
#     'alpha_phase': 1,
#     'alpha_amp': 1,
#     'grayscale': True,
#     'deterministic': False,
#     'watson_pretrained': True,
# }

# batch_sizes = [-1,[6,14],[6,13],[6,30],[7,34],[7,32]]
# for i in range(1,6):
#     for j in enumerate([True, False]):
#         parameters['architecture'] = 'ConvGRU{}'.format(i)
#         parameters['image_space_real'] = j
#         print("architecture = ", parameters['architecture'])
#         print("image_space_real = ", parameters['image_space_real'])
#         start = time.time()
#         a = ConvGRU(parameters, 1).cuda(1)
#         inp  = torch.zeros(batch_sizes[i][j],34,8,64,64,2).cuda(1)
#         outp = a(inp)
#         os.system('nvidia-smi | grep 350W | head -2 | tail -1')
#         del a
#         del outp
#         del inp
#         torch.cuda.empty_cache()
#         print("Time taken = ", time.time() - start, flush = True)
#         print('',flush = True)


# = batch_sizes[int(parameters['architecture'][-1])][parameters['image_space_real']]