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
from utils.models.gruComponents import IFFT_module, GRUImageSpaceDecoder, GRUImageSpaceEncoder, GRUKspaceModel
from utils.models.gruGates import GRUGate_KSpace, GRUGate_complex2d, GRUGate_real2d

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
        assert(isinstance(input, type([])))
        assert(isinstance(prev_state, type([])))
        stacked_input = [torch.cat([input[i],prev_state[i]], dim=1) for i in range(len(input))]
        update = self.update_gate(stacked_input, activation = 'sigmoid')
        reset = self.reset_gate(stacked_input, activation = 'sigmoid')
        stacked_input2 = [torch.cat([input[i],prev_state[i]*reset[i]], dim=1) for i in range(len(input))]
        out_input = self.out_gate(stacked_input2, activation = 'tanh')
        new_state = [(prev_state[i] * (1 - update[i])) + (out_input[i] * update[i]) for i in range(len(input))]
        return new_state

class ConvGRU(nn.Module):
    def __init__(self, parameters, n_layers = 1):
        super(ConvGRU, self).__init__()

        self.train_parameters = parameters
        self.n_layers = n_layers
        self.num_window = parameters['window_size']
        self.num_coils = parameters['num_coils']
        self.image_space_real = parameters['image_space_real']
        if parameters['architecture'] == 'ConvGRU1':
            self.ifft_m = IFFT_module(parameters)
            self.encoder_m = GRUImageSpaceEncoder(in_channels = self.num_coils, out_channels = 128, image_space_real = self.image_space_real)
            self.decoder_m = GRUImageSpaceDecoder(in_channels = 128, image_space_real = self.image_space_real)
            self.post_model = lambda x:self.decoder_m(*self.encoder_m(self.ifft_m(*x)))
            self.pre_model = lambda x:[x]
            self.gate_model = lambda :GRUGate_KSpace(parameters)
            self.hidden_chan = [parameters['num_coils']]
            self.hidden_real = False
        elif parameters['architecture'] == 'ConvGRU2':
            self.ifft_m = IFFT_module(parameters)
            self.encoder_m = GRUImageSpaceEncoder(in_channels = self.num_coils, out_channels = 128, image_space_real = self.image_space_real)
            self.decoder_m = GRUImageSpaceDecoder(in_channels = 128, image_space_real = self.image_space_real)
            self.post_model = lambda x:self.decoder_m(*self.encoder_m(self.ifft_m(*x)))
            self.kspace_m = GRUKspaceModel(input_coils = parameters['num_coils'], output_coils = parameters['num_coils'])
            self.pre_model = lambda x: [self.kspace_m(x)]
            self.gate_model = lambda : GRUGate_complex2d([2*parameters['num_coils']], [parameters['num_coils']])
            self.hidden_chan = [parameters['num_coils']]
            self.hidden_real = False
        elif parameters['architecture'] == 'ConvGRU3':
            self.kspace_m = GRUKspaceModel(input_coils = parameters['num_coils'], output_coils = parameters['num_coils'])
            self.ifft_m = IFFT_module(parameters)
            self.encoder_m = GRUImageSpaceEncoder(in_channels = self.num_coils, out_channels = 128, image_space_real = self.image_space_real)
            self.decoder_m = GRUImageSpaceDecoder(in_channels = 128, image_space_real = self.image_space_real)
            self.post_model = lambda x: self.decoder_m(*self.encoder_m(*x))
            self.pre_model = lambda x : [self.ifft_m(self.kspace_m(x))]
            if self.image_space_real:
                self.gate_model = lambda : GRUGate_real2d([2*parameters['num_coils']], [parameters['num_coils']])
            else:
                self.gate_model = lambda : GRUGate_complex2d([2*parameters['num_coils']], [parameters['num_coils']])
            self.hidden_chan = [parameters['num_coils']]
            self.hidden_real = False
        elif parameters['architecture'] == 'ConvGRU4':
            self.kspace_m = GRUKspaceModel(input_coils = parameters['num_coils'], output_coils = parameters['num_coils'])
            self.ifft_m = IFFT_module(parameters)
            self.encoder_m = GRUImageSpaceEncoder(in_channels = self.num_coils, out_channels = 128, image_space_real = self.image_space_real)
            self.decoder_m = GRUImageSpaceDecoder(in_channels = 128, image_space_real = self.image_space_real)
            self.post_model = lambda x: self.decoder_m(*x)
            self.pre_model = lambda x : self.encoder_m(self.ifft_m(self.kspace_m(x)))
            if self.image_space_real:
                self.gate_model = lambda : GRUGate_real2d([2*self.encoder_m.latent_channels[0]], [self.encoder_m.latent_channels[0]])
            else:
                self.gate_model = lambda : GRUGate_complex2d([2*self.encoder_m.latent_channels[0]], [self.encoder_m.latent_channels[0]])
            self.hidden_chan = [self.encoder_m.latent_channels[0]]
            self.hidden_real = self.image_space_real
        elif parameters['architecture'] == 'ConvGRU5':
            self.kspace_m = GRUKspaceModel(input_coils = parameters['num_coils'], output_coils = parameters['num_coils'])
            self.ifft_m = IFFT_module(parameters)
            self.encoder_m = GRUImageSpaceEncoder(in_channels = self.num_coils, out_channels = 128, image_space_real = self.image_space_real)
            self.decoder_m = GRUImageSpaceDecoder(in_channels = 128, image_space_real = self.image_space_real)
            self.post_model = lambda x: self.decoder_m(*x)
            self.pre_model = lambda x : self.encoder_m(self.ifft_m(self.kspace_m(x)))
            if self.image_space_real:
                self.gate_model = lambda : GRUGate_real2d([2*self.encoder_m.latent_channels[0]], [self.encoder_m.latent_channels[0]])
            else:
                self.gate_model = lambda : GRUGate_complex2d([2*self.encoder_m.latent_channels[0]], [self.encoder_m.latent_channels[0]])
            self.hidden_chan = [self.encoder_m.latent_channels]
            self.hidden_real = self.image_space_real
        else:
            print("Unrecognised GRU architecture mode '{}'".format(parameters['architecture']), flush = True)
            os._exit(1)

        cells = []
        for i in range(self.n_layers):
            cell = ConvGRUCell(parameters, self.gate_model)
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def get_kspace_params(self):
        if self.train_parameters['architecture'] == 'ConvGRU1':
            ans = []
            for ci in self.cells:
                ans += list(ci.parameters())
            return ans
        else:
            return self.kspace_m.parameters()

    def get_ispace_params(self):
        return list(self.decoder_m.parameters()) + list(self.encoder_m.parameters())

    def get_gate_params(self):
        if self.train_parameters['architecture'] == 'ConvGRU1':
            return []
        ans = []
        for ci in self.cells:
            ans += list(ci.parameters())
        return ans

    def forward(self, x, hidden=None):
        '''
        Input size - B, T, C, X, Y
        B = batch
        T = Video Length
        C = Num Coils
        X, Y = resolution
        '''
        B, T, C, X, Y, complex_dim = x.shape

        ans = None
        iter_outp = hidden
        preprocessed = [val.view(B,T,-1,*val.shape[2:]) for val in self.pre_model(x.view(B*T,*x.shape[2:]))]
        if iter_outp is None:
            iter_outp = []
            for i in range(len(self.hidden_chan)):
                iter_outp.append(torch.zeros((B, *preprocessed[i].shape[2:]), device = x.device))
        for ti in range(T):
            for layer_idx in range(self.n_layers):
                cell = self.cells[layer_idx]
                '''
                Input size = B, C, 1, X, Y
                Output sizes : 
                Mode1 = B, C, 1, X, Y
                Mode2 = B, C, 1, X, Y
                Mode3 = B, C, 1, X, Y
                Mode4 = B, channel, X, Y
                '''
                hidden = iter_outp
                iter_outp = cell([val[:,ti] for val in preprocessed[:len(iter_outp)]], iter_outp)
                if len(iter_outp) != len(preprocessed):
                    temp1 = iter_outp + [val[:,ti] for val in preprocessed[len(iter_outp):]]
                else:
                    temp1 = iter_outp
                if ans is None:
                    ans = temp1
                else:
                    ans = [torch.cat([x1, x2], dim=0) for x1,x2 in zip(ans, temp1)]
                
        ans = self.post_model(ans).squeeze()
        ans = ans.view(B,T,*ans.shape[1:])
        if not self.image_space_real:
            ans = (ans**2).sum(-1)**0.5

        return ans, [x.detach() for x in hidden]

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