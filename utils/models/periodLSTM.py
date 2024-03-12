import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms
import numpy as np
import os
import sys
import time
import kornia
import matplotlib.pyplot as plt
import utils.models.complexCNNs.cmplx_conv as cmplx_conv
import utils.models.complexCNNs.cmplx_dropout as cmplx_dropout
import utils.models.complexCNNs.cmplx_upsample as cmplx_upsample
import utils.models.complexCNNs.cmplx_activation as cmplx_activation
import utils.models.complexCNNs.radial_bn as radial_bn

import sys
sys.path.append('/root/Cardiac-MRI-Reconstrucion/')

from utils.functions import fetch_loss_function

EPS = 1e-10
CEPS = torch.complex(torch.tensor(EPS),torch.tensor(EPS))

class Identity(nn.Module):
    def __init__(self, a, b):
        super(Identity, self).__init__()
        self.m = nn.Linear(3,3)

    def forward(self, x):
        return x

def mylog(x,base = 10):
    return x.log10()/torch.tensor(base).log10()

def gaussian_2d(shape, sigma=None):
    """
    Generate a 2D Gaussian mask.

    Parameters:
        shape (tuple): Shape of the output array (height, width).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: 2D Gaussian mask.
    """
    if sigma is None:
        sigma = shape[0]//5
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sum_h = h.sum()
    if sum_h != 0:
        h /= sum_h
    h = h - h.min()
    h = h/(h.max()+1e-10)
    h *= 9
    h += 1
    return h

def special_trim(x):
    percentile_95 = np.percentile(x.detach().cpu(), 95)
    percentile_5 = np.percentile(x.detach().cpu(), 5)
    x = x.clip(percentile_5, percentile_95)
    x = x - x.min().detach()
    x = x/ (x.max().detach() + EPS)
    return x

def fetch_lstm_type(parameters):
    ispace_name = parameters['ispace_architecture']
    kspace_name = parameters['kspace_architecture']
    if ispace_name == 'ILSTM1':
        im = ImageSpaceModel1
    elif ispace_name == 'ILSTM2':
        im = ImageSpaceModel2
    elif ispace_name == 'ILSTM3':
        im = convLSTM_Ispace1

    if kspace_name == 'KLSTM1':
        km = convLSTM_Kspace1
    if kspace_name == 'KLSTM2':
        km = convLSTM_Kspace2
    return km, im

class concatConv(nn.Module):
    def __init__(self, cnn_func, relu_func, gate_input_size = 8, hidden_channels = 32, gate_output_size = 1, n_layers = 4, catmode = True):
        super(concatConv, self).__init__()
        self.layerlist = []
        self.catmode = catmode
        self.relu_func = relu_func()
        self.n_layers = n_layers
        if self.n_layers == 1:
            self.layerlist.append(cnn_func(gate_input_size, gate_output_size, (1,1), stride = (1,1), padding = (0,0)))
        else:
            self.layerlist.append(cnn_func(gate_input_size, hidden_channels, (3,3), stride = (1,1), padding = (1,1)))
            inlen = hidden_channels
            if self.catmode:
                skiplen = gate_input_size
            else:
                skiplen = 0
            for i in range(n_layers-2):
                self.layerlist.append(cnn_func(inlen+skiplen, hidden_channels, (3,3), stride = (1,1), padding = (1,1)))
                inlen = hidden_channels
            self.layerlist.append(cnn_func(inlen+skiplen, gate_output_size, (1,1), stride = (1,1), padding = (0,0)))

        self.layerlist = nn.ModuleList(self.layerlist)

    def forward(self, x):
        if self.n_layers == 1:
            return self.layerlist[0](x)

        to_cat = x.clone()
        curr_output = self.layerlist[0](x)
        
        for i,layer in enumerate(self.layerlist[1:self.n_layers-1]):
            input = torch.cat((curr_output, to_cat), 1)
            curr_output = self.relu_func(layer(input))


        input = torch.cat((curr_output, to_cat), 1)
        return self.layerlist[-1](input)



class convLSTMcell_kspace(nn.Module):
    def __init__(self, history_length = 1, num_coils = 8, phase_tanh_mode = False, sigmoid_mode = True, phase_real_mode = False, phase_theta = False, forget_gate_coupled = False, forget_gate_same_coils = False, forget_gate_same_phase_mag = False, lstm_input_mask = False, catmode = False, n_layers = 4, n_hidden = 16, n_lstm_cells = 1, coilwise = False, kspace_combine_coils = False):
        super(convLSTMcell_kspace, self).__init__()
        self.phase_tanh_mode = phase_tanh_mode
        self.n_lstm_cells = n_lstm_cells
        self.sigmoid_mode = sigmoid_mode
        self.n_hidden = n_hidden
        self.history_length = history_length
        self.num_coils = num_coils
        self.catmode = catmode
        self.kspace_combine_coils = kspace_combine_coils
        self.coilwise = coilwise
        self.phase_real_mode = phase_real_mode
        self.phase_theta = phase_theta
        self.forget_gate_coupled = forget_gate_coupled
        self.forget_gate_same_coils = forget_gate_same_coils
        self.forget_gate_same_phase_mag = forget_gate_same_phase_mag
        self.lstm_input_mask = lstm_input_mask
        
        mag_cnn_func = nn.Conv2d
        mag_relu_func = nn.ReLU
        
        if phase_real_mode:
            phase_cnn_func = nn.Conv2d
            phase_relu_func = nn.ReLU
        else:
            phase_cnn_func = cmplx_conv.ComplexConv2d
            phase_relu_func = cmplx_activation.CReLU

        if self.phase_theta:
            if self.phase_tanh_mode:
                self.phase_activation = lambda x: np.pi*torch.tanh(x)
            else:
                self.phase_activation = lambda x: x
        else:
            if self.phase_tanh_mode:
                self.phase_activation = lambda x: torch.tanh(x)
            else:
                self.phase_activation = lambda x: x

        if self.kspace_combine_coils:
            self.input_gate_output_size = 1
        else:
            self.input_gate_output_size = self.num_coils

        if self.coilwise:
            gate_input_size = 1 + ((1 + self.history_length))
            self.input_gate_output_size = 1
        else:
            gate_input_size = self.input_gate_output_size + ((1 + self.history_length)*self.num_coils)

        if self.lstm_input_mask:
            gate_input_size += 1

        hidden_channels = self.n_hidden

        if self.forget_gate_same_coils:
            forget_gate_output_size = 1
        else:
            if self.kspace_combine_coils:
                forget_gate_output_size = 1
            else:
                forget_gate_output_size = self.num_coils

        self.mag_inputGates = nn.ModuleList([concatConv(mag_cnn_func, mag_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, catmode = self.catmode) for i in range(self.n_lstm_cells)])
        if not self.forget_gate_same_phase_mag:
            self.phase_inputGates = nn.ModuleList([concatConv(phase_cnn_func, phase_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, catmode = self.catmode) for i in range(self.n_lstm_cells)])
        if not self.forget_gate_coupled:
            self.mag_forgetGates = nn.ModuleList([concatConv(mag_cnn_func, mag_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, catmode = self.catmode) for i in range(self.n_lstm_cells)])
            if not self.forget_gate_same_phase_mag:
                self.phase_forgetGates = nn.ModuleList([concatConv(phase_cnn_func, phase_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, catmode = self.catmode) for i in range(self.n_lstm_cells)])
            self.mag_outputGates = nn.ModuleList([concatConv(mag_cnn_func, mag_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, catmode = self.catmode) for i in range(self.n_lstm_cells)])
            self.phase_outputGates = nn.ModuleList([concatConv(phase_cnn_func, phase_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, catmode = self.catmode) for i in range(self.n_lstm_cells)])
        self.mag_inputProcs = nn.ModuleList([concatConv(mag_cnn_func, mag_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, catmode = self.catmode) for i in range(self.n_lstm_cells)])
        self.phase_inputProcs = nn.ModuleList([concatConv(phase_cnn_func, phase_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, catmode = self.catmode) for i in range(self.n_lstm_cells)])
        
        # self.decoder2 = nn.Linear(64*64*2, 64*64)

    def forward(self, hist_mag, hist_phase, gt_mask = None, mag_prev_states = None, mag_prev_outputs = None, phase_prev_states = None, phase_prev_outputs = None):
        # x is a batch of video frames at a single time stamp

        if mag_prev_states is None:
            if self.coilwise:
                mag_shape1 = (hist_mag.shape[0], self.num_coils, *hist_mag.shape[2:])
                phase_shape1 = (hist_phase.shape[0], self.num_coils, *hist_phase.shape[2:])
            else:
                mag_shape1 = (hist_mag.shape[0], self.input_gate_output_size, *hist_mag.shape[2:])
                phase_shape1 = (hist_phase.shape[0], self.input_gate_output_size, *hist_phase.shape[2:])
            mag_prev_states = [torch.zeros(mag_shape1, device = hist_mag.device) for _ in range(self.n_lstm_cells)]
            mag_prev_outputs = [torch.zeros(mag_shape1, device = hist_mag.device) for _ in range(self.n_lstm_cells)]
            phase_prev_states = [torch.zeros(phase_shape1, device = hist_phase.device) for _ in range(self.n_lstm_cells)]
            phase_prev_outputs = [torch.zeros(phase_shape1, device = hist_phase.device) for _ in range(self.n_lstm_cells)]


        og_B, og_C, _,_ = hist_mag.shape
        if self.coilwise:
            hist_mag = hist_mag.reshape(og_B*og_C, 1, *hist_mag.shape[2:])
            hist_phase = hist_phase.reshape(og_B*og_C, 1, *hist_phase.shape[2:])
            gt_mask = gt_mask.repeat((1, self.num_coils,1,1)).reshape(og_B*og_C, 1, *hist_phase.shape[2:])
            for i_cell in range(self.n_lstm_cells):
                mag_prev_states[i_cell] = mag_prev_states[i_cell].reshape(og_B*og_C, 1, *mag_prev_states[i_cell].shape[2:])
                phase_prev_states[i_cell] = phase_prev_states[i_cell].reshape(og_B*og_C, 1, *phase_prev_states[i_cell].shape[2:])
                mag_prev_outputs[i_cell] = mag_prev_outputs[i_cell].reshape(og_B*og_C, 1, *mag_prev_outputs[i_cell].shape[2:])
                phase_prev_outputs[i_cell] = phase_prev_outputs[i_cell].reshape(og_B*og_C, 1, *phase_prev_outputs[i_cell].shape[2:])

        new_mag_outputs = [hist_mag]
        new_phase_outputs = [hist_phase]
        new_mag_states = []
        new_phase_states = []
        for i_cell in range(self.n_lstm_cells):
            if self.lstm_input_mask:
                mag_inp_cat = torch.cat((new_mag_outputs[i_cell], mag_prev_outputs[i_cell], ((gt_mask*2.)-1.).type(torch.float)), 1)
                phase_inp_cat = torch.cat((new_phase_outputs[i_cell], phase_prev_outputs[i_cell], ((gt_mask*2.)-1.).type(torch.float)), 1)
            else:
                mag_inp_cat = torch.cat((new_mag_outputs[i_cell], mag_prev_outputs[i_cell]), 1)
                phase_inp_cat = torch.cat((new_phase_outputs[i_cell], phase_prev_outputs[i_cell]), 1)

            assert(self.sigmoid_mode)
            mag_it = torch.sigmoid(self.mag_inputGates[i_cell](mag_inp_cat))
            if not self.forget_gate_same_phase_mag:
                phase_it = torch.sigmoid(self.phase_inputGates[i_cell](phase_inp_cat))
            else:
                phase_it = mag_it

            if not self.forget_gate_coupled:
                mag_ft = torch.sigmoid(self.mag_forgetGates[i_cell](mag_inp_cat))
                if not self.forget_gate_same_phase_mag:
                    phase_ft = torch.sigmoid(self.phase_forgetGates[i_cell](phase_inp_cat))
                else:
                    phase_ft = mag_ft
                mag_ot = torch.sigmoid(self.mag_outputGates[i_cell](mag_inp_cat))
                phase_ot = torch.sigmoid(self.phase_outputGates[i_cell](phase_inp_cat))
            else:
                mag_ft = 1 - mag_it
                phase_ft = 1 - phase_it
                mag_ot = torch.ones_like(mag_ft, device = mag_ft.device)
                phase_ot = torch.ones_like(phase_ft, device = phase_ft.device)

            if self.forget_gate_same_coils:
                mag_ft = mag_ft.repeat(1,self.input_gate_output_size,1,1)
                phase_ft = phase_ft.repeat(1,self.input_gate_output_size,1,1)
                mag_it = mag_it.repeat(1,self.input_gate_output_size,1,1)
                phase_it = phase_it.repeat(1,self.input_gate_output_size,1,1)

            mag_Cthat = self.mag_inputProcs[i_cell](mag_inp_cat)
            phase_Cthat = self.phase_activation(self.phase_inputProcs[i_cell](phase_inp_cat))


            new_mag_states.append((mag_ft * mag_prev_states[i_cell]) + (mag_it * mag_Cthat))
            new_phase_states.append((phase_ft * phase_prev_states[i_cell]) + (phase_it * phase_Cthat))

            new_mag_outputs.append(new_mag_states[i_cell]*mag_ot)
            new_phase_outputs.append(self.phase_activation(new_phase_states[i_cell])*phase_ot)


        if self.coilwise:
            for i_cell in range(self.n_lstm_cells):
                new_mag_states[i_cell] = new_mag_states[i_cell].reshape(og_B,og_C,*new_mag_states[i_cell].shape[2:])
                new_phase_states[i_cell] = new_phase_states[i_cell].reshape(og_B,og_C,*new_phase_states[i_cell].shape[2:])
                new_mag_outputs[i_cell+1] = new_mag_outputs[i_cell+1].reshape(og_B,og_C,*new_mag_outputs[i_cell+1].shape[2:])
                new_phase_outputs[i_cell+1] = new_phase_outputs[i_cell+1].reshape(og_B,og_C,*new_phase_outputs[i_cell+1].shape[2:])

        return new_mag_states, new_phase_states, new_mag_outputs[1:], new_phase_outputs[1:]

class convLSTMcell(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, tanh_mode = False, sigmoid_mode = True, real_mode = False, theta = False, mini = False):
        super(convLSTMcell, self).__init__()
        self.tanh_mode = tanh_mode
        self.mini = mini
        self.sigmoid_mode = sigmoid_mode
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.real_mode = real_mode
        self.theta = theta
        if real_mode:
            cnn_func = nn.Conv2d
            relu_func = nn.ReLU
        else:
            cnn_func = cmplx_conv.ComplexConv2d
            relu_func = cmplx_activation.CReLU

        if theta:
            if self.tanh_mode:
                self.activation = lambda x: np.pi*torch.tanh(x)
            else:
                self.activation = lambda x: x
        else:
            if self.tanh_mode:
                self.activation = lambda x: torch.tanh(x)
            else:
                self.activation = lambda x: x

        if not mini:
            self.inputGate = nn.Sequential(
                    cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, self.out_channels, (1,1), stride = (1,1), padding = (0,0)),
                    # relu_func(),
                )
            self.forgetGate = nn.Sequential(
                    cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, self.out_channels, (1,1), stride = (1,1), padding = (0,0)),
                    # relu_func(),
                )
            self.outputGate = nn.Sequential(
                    cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, self.out_channels, (1,1), stride = (1,1), padding = (0,0)),
                    # relu_func(),
                )
            self.inputProc = nn.Sequential(
                    cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, self.out_channels, (1,1), stride = (1,1), padding = (0,0)),
                    # relu_func(),
                )
        else:
            self.inputGate = nn.Sequential(
                    cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, self.out_channels, (1,1), stride = (1,1), padding = (0,0)),
                    # relu_func(),
                )
            self.forgetGate = nn.Sequential(
                    cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, self.out_channels, (1,1), stride = (1,1), padding = (0,0)),
                    # relu_func(),
                )
            self.outputGate = nn.Sequential(
                    cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, self.out_channels, (1,1), stride = (1,1), padding = (0,0)),
                    # relu_func(),
                )
            self.inputProc = nn.Sequential(
                    cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                    relu_func(),
                    cnn_func(2*self.out_channels, self.out_channels, (1,1), stride = (1,1), padding = (0,0)),
                    # relu_func(),
                )

        # self.decoder2 = nn.Linear(64*64*2, 64*64)

    def forward(self, x, prev_state = None, prev_output = None):
        # x is a batch of video frames at a single time stamp
        if prev_state is None:
            shape1 = (x.shape[0], self.out_channels, *x.shape[2:])
            prev_state = torch.zeros(shape1, device = x.device)
            prev_output = torch.zeros(shape1, device = x.device)
        inp_cat = torch.cat((x, prev_output), 1)

        if self.sigmoid_mode:
            ft = torch.sigmoid(self.forgetGate(inp_cat))
            it = torch.sigmoid(self.inputGate(inp_cat))
            ot = torch.sigmoid(self.outputGate(inp_cat))
        else:
            ft = self.forgetGate(inp_cat)
            it = self.inputGate(inp_cat)
            ot = self.outputGate(inp_cat)
        #### DEBUG - remove tanh if doesnt work here
        Cthat = self.activation(self.inputProc(inp_cat))
        Ct_new = (ft * prev_state) + (it * Cthat)
        ht = self.activation(Ct_new)*ot
        # print(ht.min())
        # print(ht.max())
        # print(x.min())
        # print(x.max())
        # print('------------------------------------------')
            # ht = ht + x

        return Ct_new, ht

class convLSTM_Kspace1(nn.Module):
    def __init__(self, parameters, proc_device, sigmoid_mode = True, two_cell = False):
        super(convLSTM_Kspace1, self).__init__()
        self.param_dic = parameters
        self.history_length = self.param_dic['history_length']
        self.n_coils = self.param_dic['num_coils']


        if self.param_dic['kspace_predict_mode'] == 'thetas':
            theta = True
            self.real_mode = True
        elif self.param_dic['kspace_predict_mode'] == 'cosine':
            theta = False
            self.real_mode = True
            assert self.param_dic['kspace_tanh'], 'Tanh must be applied when predicting only cosines'
        elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
            theta = False
            self.real_mode = False
        else:
            assert 0

        self.kspace_m = convLSTMcell_kspace(
                    phase_tanh_mode = self.param_dic['kspace_tanh'], 
                    sigmoid_mode = sigmoid_mode, 
                    phase_real_mode = self.real_mode, 
                    num_coils = self.n_coils,
                    history_length = self.history_length,
                    phase_theta = theta, 
                    forget_gate_coupled = self.param_dic['forget_gate_coupled'],
                    forget_gate_same_coils = self.param_dic['forget_gate_same_coils'],
                    forget_gate_same_phase_mag = self.param_dic['forget_gate_same_phase_mag'],
                    lstm_input_mask = self.param_dic['lstm_input_mask'],
                    catmode = self.param_dic['concat'],
                    n_layers = self.param_dic['n_layers'],
                    n_hidden = self.param_dic['n_hidden'],
                    n_lstm_cells = self.param_dic['n_lstm_cells'],
                    kspace_combine_coils = self.param_dic['kspace_combine_coils'],
                    coilwise = self.param_dic['coilwise'],
                )

        if self.param_dic['ispace_lstm']:
            self.ispacem = convLSTMcell(
                    tanh_mode = True, 
                    sigmoid_mode = sigmoid_mode, 
                    real_mode = True, 
                    in_channels = 1, 
                    out_channels = 1, 
                )
        self.SSIM = kornia.metrics.SSIM(11)

        if self.param_dic['crop_loss']:
            mask = gaussian_2d((self.param_dic['image_resolution'],self.param_dic['image_resolution'])).reshape(1,1,self.param_dic['image_resolution'],self.param_dic['image_resolution'])

        else:
            mask = np.ones((1,1,self.param_dic['image_resolution'],self.param_dic['image_resolution']))

        self.lossmask = torch.FloatTensor(mask).to(proc_device)

        mask = torch.FloatTensor(gaussian_2d((self.param_dic['image_resolution'],self.param_dic['image_resolution']), sigma = self.param_dic['image_resolution']//10))
        mask = torch.fft.fftshift(mask)
        mask = mask - mask.min()
        mask = mask / (mask.max() + EPS)
        mask = (1-mask).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.predr_mask = torch.FloatTensor(mask)

        assert(sigmoid_mode)

    # def time_analysis(self, fft_exp, device, periods, ispace_model):
    #     times = []
    #     with torch.no_grad():
    #         # mag_log = (fft_exp.abs()+EPS).log()
    #         # phase = fft_exp / (mag_log.exp())
    #         # phase = torch.stack((phase.real, phase.imag), -1)

    #         prev_state1 = None
    #         prev_output1 = None
    #         prev_state2 = None
    #         prev_output2 = None
    #         prev_state3 = None
    #         prev_output3 = None

    #         predr = torch.zeros(fft_exp.shape)
 
    #         for ti in range(fft_exp.shape[1]):
    #             start1 = time.time()
    #             hist_ind = (torch.arange(self.history_length+1).repeat(fft_exp.shape[0],1) - self.history_length)
    #             hist_ind = hist_ind * periods.reshape(-1,1).cpu()
    #             hist_ind += ti
    #             temp1 = hist_ind.clone()
    #             temp1[temp1 < 0] = 9999999999
    #             min_vals = temp1.min(1)[0]
    #             base = torch.zeros(temp1.shape) + min_vals.reshape(-1,1)

    #             hist_ind = hist_ind - base
    #             hist_ind[hist_ind < 0] = 0
    #             hist_ind = (hist_ind + base).long()

    #             mult = (torch.arange(fft_exp.shape[0])*fft_exp.shape[1]).reshape(-1,1)
    #             hist_ind = hist_ind + mult
    #             hist_fft = fft_exp.reshape(-1, *fft_exp.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *fft_exp.shape[3:]).to(device)
    #             hist_mag = mylog((hist_fft.abs()+EPS), base = self.param_dic['logarithm_base'])
    #             hist_phase = hist_fft / (self.param_dic['logarithm_base']**hist_mag)
    #             hist_phase = torch.stack((hist_phase.real, hist_phase.imag), -1)

    #             if self.param_dic['kspace_predict_mode'] == 'thetas':
    #                 hist_phase = torch.atan2(hist_phase[:,:,:,:,1],hist_phase[:,:,:,:,0])
    #             elif self.param_dic['kspace_predict_mode'] == 'cosine':
    #                 hist_phase = hist_phase[:,:,:,:,0]
    #             elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
    #                 hist_phase = hist_phase
    #             else:
    #                 assert 0

    #             prev_state2, prev_state1, prev_output2, prev_output1 = self.kspace_m(hist_mag, hist_phase, prev_state2, prev_output2, prev_state1, prev_output1)

    #             if self.param_dic['kspace_predict_mode'] == 'cosine':
    #                 phase_ti = torch.complex(prev_output1, ((1-(prev_output1**2)) + EPS)**0.5)
    #             elif self.param_dic['kspace_predict_mode'] == 'thetas':
    #                 phase_ti = torch.complex(torch.cos(prev_output1), torch.sin(prev_output1))
    #             elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
    #                 # prev_output1 = prev_output1 / (((prev_output1**2).sum(-1)+EPS)**0.5).unsqueeze(-1).detach()
    #                 phase_ti = torch.complex(prev_output1[:,:,:,:,0], prev_output1[:,:,:,:,1])

    #             predr_ti = torch.fft.ifft2(torch.fft.ifftshift((self.param_dic['logarithm_base']**prev_output2)*phase_ti, dim = (-2,-1))).real.clip(-200,200)
    #             if self.param_dic['ispace_lstm']:
    #                 prev_state3, prev_output3 = self.ispacem(predr_ti, prev_state3, prev_output3)
    #             else:
    #                 prev_output3 = ispace_model(predr_ti)
                
    #             if self.param_dic['ispace_lstm']:
    #                 prev_output3 = prev_output3.detach()            
    #             prev_output2 = prev_output2.detach()
    #             prev_output1 = prev_output1.detach()            

    #             predr[:,ti,:,:,:] = prev_output3.cpu()
    #             times.append(time.time() - start1)

    #     return times


    def forward(self, fft_exp, gt_masks = None, device = torch.device('cpu'), periods = None, targ_phase = None, targ_mag_log = None, targ_real = None, og_video = None):

        mag_log = fft_exp.abs()
        phase = fft_exp / (self.param_dic['logarithm_base']**mag_log)
        mag_log = mylog((mag_log+torch.tensor(1e-4)).clip(1e-4, 1e20), base = self.param_dic['logarithm_base'])
        mag_log = mag_log + 4

        del fft_exp

        phase = torch.stack((phase.real, phase.imag), -1)

        prev_states1 = None
        prev_outputs1 = None
        prev_states2 = None
        prev_outputs2 = None
        prev_state3 = None
        prev_output3 = None

        if self.param_dic['kspace_combine_coils']:
            ans_coils = 1
        else:
            ans_coils = self.param_dic['num_coils']

        ans_mag_log = torch.zeros(*mag_log.shape[0:2], ans_coils, *mag_log.shape[3:])
        length = ans_mag_log.shape[-1]
        x_size, y_size = length, length
        x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
        cell = (length//2, length//2)
        dists = ((x_arr - cell[0])**2 + (y_arr - cell[1])**2)**0.33
        dists = torch.FloatTensor(dists+1).to(device)
        ans_phase = torch.zeros(*phase.shape[0:2], ans_coils, *phase.shape[3:])
        predr = torch.zeros(*mag_log.shape[0:2], ans_coils, *mag_log.shape[3:])

        if targ_phase is not None:
            loss_phase = 0
            loss_mag = 0
            loss_real = 0
            loss_l1 = 0
            loss_l2 = 0
            loss_ss1 = 0
            criterionL1 = nn.L1Loss().to(device)
            criterionCos = nn.CosineSimilarity(dim = 4) 
        else:
            loss_phase = None
            loss_mag = None
            loss_real = None
            loss_l1 = 0
            loss_l2 = 0
            loss_ss1 = 0

        for ti in range(mag_log.shape[1]):
            if periods is None:
                hist_phase = phase[:,ti]
                hist_mag = mag_log[:,ti]
            else:
                hist_ind = (torch.arange(self.history_length+1).repeat(mag_log.shape[0],1) - self.history_length)
                hist_ind = hist_ind * periods.reshape(-1,1).cpu()
                hist_ind += ti
                temp1 = hist_ind.clone()
                temp1[temp1 < 0] = 9999999999
                min_vals = temp1.min(1)[0]
                base = torch.zeros(temp1.shape) + min_vals.reshape(-1,1)

                hist_ind = hist_ind - base
                hist_ind[hist_ind < 0] = 0
                hist_ind = (hist_ind + base).long()

                mult = (torch.arange(mag_log.shape[0])*mag_log.shape[1]).reshape(-1,1)
                hist_ind = hist_ind + mult      

                hist_phase = phase.reshape(-1, *phase.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *phase.shape[3:])
                hist_mag = mag_log.reshape(-1, *mag_log.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *mag_log.shape[3:])
                # hist_mask = gt_masks.reshape(-1, *gt_masks.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *gt_masks.shape[3:])

            if self.param_dic['kspace_predict_mode'] == 'thetas':
                hist_phase = torch.atan2(hist_phase[:,:,:,:,1],hist_phase[:,:,:,:,0]) + 4
            elif self.param_dic['kspace_predict_mode'] == 'cosine':
                hist_phase = hist_phase[:,:,:,:,0]
            elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                hist_phase = hist_phase
            else:
                assert 0

            if prev_outputs2 is not None:
                prev_outputs2 = [x + 4 for x in prev_outputs2]
                prev_outputs1 = [x + 4 for x in prev_outputs1]
            prev_states2, prev_states1, prev_outputs2, prev_outputs1 = self.kspace_m(hist_mag, hist_phase, gt_masks[:,ti,:,:,:], prev_states2, prev_outputs2, prev_states1, prev_outputs1)
            prev_outputs2 = [x - 4 for x in prev_outputs2]
            prev_outputs1 = [x - 4 for x in prev_outputs1]

            del hist_mag
            del hist_phase
            
            for i_cell in range(self.param_dic['n_lstm_cells']):
                if self.param_dic['kspace_predict_mode'] == 'cosine':
                    phase_ti = torch.complex(prev_outputs1[i_cell], ((1-(prev_outputs1[i_cell]**2)) + EPS)**0.5)
                elif self.param_dic['kspace_predict_mode'] == 'thetas':
                    phase_ti = torch.complex(torch.cos(prev_outputs1[i_cell]), torch.sin(prev_outputs1[i_cell]))
                elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                    prev_outputs1[i_cell] = prev_outputs1[i_cell] / (((prev_outputs1[i_cell]**2).sum(-1)+EPS)**0.5).unsqueeze(-1).detach()
                    phase_ti = torch.complex(prev_outputs1[i_cell][:,:,:,:,0], prev_outputs1[i_cell][:,:,:,:,1])

                stacked_phase = torch.stack((phase_ti.real, phase_ti.imag), -1)

                lstm_predicted_fft = (self.param_dic['logarithm_base']**prev_outputs2[i_cell])*phase_ti
                # lstm_predicted_fft.real = lstm_predicted_fft.real - lstm_predicted_fft.real.reshape(*lstm_predicted_fft.shape[:2], -1).mean(2).unsqueeze(-1).unsqueeze(-1).detach()
                # lstm_predicted_fft.imag = lstm_predicted_fft.imag - lstm_predicted_fft.imag.reshape(*lstm_predicted_fft.shape[:2], -1).mean(2).unsqueeze(-1).unsqueeze(-1).detach()
                predr_ti = torch.fft.ifft2(torch.fft.ifftshift(lstm_predicted_fft, dim = (-2,-1)))
                predr_ti = predr_ti.abs()
                # predr_ti = predr_ti - predr_ti.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(2).detach()
                # predr_ti = predr_ti / (EPS + predr_ti.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(2).detach())

                if self.param_dic['ispace_lstm']:
                    B,C,numr, numc = predr_ti.shape
                    predr_ti = predr_ti.reshape(B*C, 1, numr, numc)
                    if prev_output3 is not None:
                        prev_output3 = prev_output3.reshape(B*C, 1, numr, numc)
                    prev_state3, prev_output3 = self.ispacem(predr_ti, prev_state3, prev_output3)
                    prev_output3 = prev_output3.reshape(B, C, numr, numc)
                    predr_ti = predr_ti.reshape(B, C, numr, numc)
                else:
                    prev_output3 = None
                
                


                # prev_output3 = prev_output3 - prev_output3.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(2).detach()
                # prev_output3 = prev_output3 / (EPS + prev_output3.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(2).detach())
                
                
                if ti >= self.param_dic['init_skip_frames']:
                    if targ_phase is not None:
                        if not self.param_dic['end-to-end-supervision']:
                            loss_mag += criterionL1((prev_outputs2[i_cell]*dists), (targ_mag_log[:,ti,:,:,:].to(device)*dists))/(mag_log.shape[1]*self.param_dic['n_lstm_cells'])
                            if self.param_dic['loss_phase'] == 'L1':
                                loss_phase += criterionL1(stacked_phase, (targ_phase[:,ti,:,:,:,:].to(device)))/(mag_log.shape[1]*self.param_dic['n_lstm_cells'])
                            elif self.param_dic['loss_phase'] == 'Cosine':
                                loss_phase += (1 - criterionCos(stacked_phase, (targ_phase[:,ti,:,:,:,:]).to(device))).mean()/(mag_log.shape[1]*self.param_dic['n_lstm_cells'])
                            elif self.param_dic['loss_phase'] == 'raw_L1':
                                if self.param_dic['kspace_predict_mode'] == 'cosine':
                                    loss_phase += criterionL1(prev_outputs1[-1], (targ_phase[:,ti,:,:,:,0]).to(device))/(mag_log.shape[1]*self.param_dic['n_lstm_cells'])
                                elif self.param_dic['kspace_predict_mode'] == 'thetas':
                                    targ_angles = torch.atan2((targ_phase[:,ti,:,:,:,1]),(targ_phase[:,ti,:,:,:,0])).to(device)
                                    loss_phase += criterionL1(prev_outputs1[-1], targ_angles)/(mag_log.shape[1]*self.param_dic['n_lstm_cells'])
                                elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                                    loss_phase += criterionL1(stacked_phase, (targ_phase[:,ti,:,:,:,:]).to(device))/(mag_log.shape[1]*self.param_dic['n_lstm_cells'])
                                else:
                                    assert 0
                            else:
                                assert 0
                            
                            targ_now = targ_real[:,ti,:,:,:].to(device)
                            targ_now = targ_now - targ_now.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(2).detach()
                            targ_now = targ_now / (EPS + targ_now.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(2).detach())


                            loss_real += criterionL1(predr_ti*self.lossmask, targ_now*self.lossmask)/(mag_log.shape[1]*self.param_dic['n_lstm_cells'])
                            if prev_output3 is not None:
                                loss_real += criterionL1(prev_output3*self.lossmask, targ_now*self.lossmask)/(mag_log.shape[1]*self.param_dic['n_lstm_cells'])
                    
                        with torch.no_grad():
                            loss_l1 += (predr_ti- targ_now).reshape(predr_ti.shape[0]*predr_ti.shape[1], -1).abs().mean(1).sum().detach().cpu()/self.param_dic['n_lstm_cells']
                            loss_l2 += (((predr_ti- targ_now).reshape(predr_ti.shape[0]*predr_ti.shape[1], -1) ** 2).mean(1).sum()).detach().cpu()/self.param_dic['n_lstm_cells']
                            ss1 = self.SSIM(predr_ti.reshape(predr_ti.shape[0]*predr_ti.shape[1],1,self.param_dic['image_resolution'],self.param_dic['image_resolution']), targ_now.reshape(prev_output3.shape[0]*prev_output3.shape[1],1,self.param_dic['image_resolution'],self.param_dic['image_resolution']))
                            ss1 = ss1.reshape(ss1.shape[0],-1)
                            loss_ss1 += ss1.mean(1).sum().detach().cpu() / (self.param_dic['n_lstm_cells'])

            
            for i_cell in range(self.param_dic['n_lstm_cells']):
                prev_outputs1[i_cell] = prev_outputs1[i_cell].detach()
                prev_outputs2[i_cell] = prev_outputs2[i_cell].detach()

            if self.param_dic['ispace_lstm']:
                prev_output3 = prev_output3.detach()

            ans_mag_log[:,ti,:,:] = prev_outputs2[-1].detach()
            ans_phase[:,ti,:,:,:,:] = stacked_phase.detach()
            if self.param_dic['ispace_lstm']:
                predr[:,ti,:,:] = prev_output3.detach()
            else:
                predr[:,ti,:,:] = predr_ti.detach()

        predr = predr * self.predr_mask

        return predr, ans_phase, ans_mag_log, loss_mag, loss_phase, loss_real, (loss_l1, loss_l2, loss_ss1)


class CoupledDown(nn.Module):
    def __init__(self, inp_channel, outp_channels):
        super(CoupledDown, self).__init__()
        ls = []
        prev = inp_channel
        for i in outp_channels:
            ls.append(cmplx_conv.ComplexConv2d(prev, i, (3,3), stride = (1,1), padding = (1,1), bias = False))
            ls.append(cmplx_activation.CReLU())
            ls.append(radial_bn.RadialBatchNorm2d(i))
            prev = i
        self.model = nn.Sequential(*ls)
        self.final = cmplx_conv.ComplexConv2d(prev, prev, (2,2), stride = (2,2), padding = (0,0))
        self.train_mode = True

    def forward(self, x):
        if self.train_mode:
            x1 = self.model(x)
        else:
            x1 = no_bn_forward(self.model, x)
        return x1, self.final(x1)

    def train_mode_set(self, bool = True):
        self.train_mode = bool

class CoupledUp(nn.Module):
    def __init__(self, inp_channel, outp_channels):
        super(CoupledUp, self).__init__()
        ls = []
        prev = inp_channel
        for i in outp_channels:
            ls.append(cmplx_conv.ComplexConv2d(prev, i, (3,3), stride = (1,1), padding = (1,1), bias = False))
            ls.append(cmplx_activation.CReLU())
            ls.append(radial_bn.RadialBatchNorm2d(i))
            prev = i
        self.model = nn.Sequential(*ls)
        self.final = cmplx_upsample.ComplexUpsample(scale_factor = 2, mode = 'bilinear')
        self.train_mode = True

    def forward(self, x):
        if self.train_mode:
            x1 = self.model(x)
        else:
            x1 = no_bn_forward(self.model, x)
        return self.final(x1)

    def train_mode_set(self, bool = True):
        self.train_mode = bool

class CoupledDownReal(nn.Module):
    def __init__(self, inp_channel, outp_channels):
        super(CoupledDownReal, self).__init__()
        ls = []
        prev = inp_channel
        for i in outp_channels:
            ls.append(nn.Conv2d(prev, i, (3,3), stride = (1,1), padding = (1,1), bias = False))
            ls.append(nn.ReLU())
            ls.append(nn.BatchNorm2d(i))
            prev = i
        self.model = nn.Sequential(*ls)
        self.final = nn.Conv2d(prev, prev, (2,2), stride = (2,2), padding = (0,0))
        self.train_mode = True

    def forward(self, x):
        if self.train_mode:
            x1 = self.model(x)
        else:
            x1 = no_bn_forward(self.model, x)
        return x1, self.final(x1)

    def train_mode_set(self, bool = True):
        self.train_mode = bool

class CoupledUpReal(nn.Module):
    def __init__(self, inp_channel, outp_channels):
        super(CoupledUpReal, self).__init__()
        ls = []
        prev = inp_channel
        for i in outp_channels:
            ls.append(nn.Conv2d(prev, i, (3,3), stride = (1,1), padding = (1,1), bias = False))
            ls.append(nn.ReLU())
            ls.append(nn.BatchNorm2d(i))
            prev = i
        self.model = nn.Sequential(*ls)
        self.final = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        self.train_mode = True

    def forward(self, x):
        if self.train_mode:
            x1 = self.model(x)
        else:
            x1 = no_bn_forward(self.model, x)
        return self.final(x1)

    def train_mode_set(self, bool = True):
        self.train_mode = bool

# class ImageSpaceModel2(nn.Module):
#     def __init__(self, parameters, proc_device):
#         super(ImageSpaceModel2, self).__init__()
#         self.param_dic = parameters
#         self.image_space_real = self.param_dic['image_space_real']
#         self.num_coils = self.param_dic['num_coils']
#         if self.image_space_real:
#             self.down1 = CoupledDownReal(1, [32,32])
#             self.down2 = CoupledDownReal(32, [64,64])
#             self.down3 = CoupledDownReal(64, [128,128])
#             self.up1 = CoupledUpReal(128, [256,128])
#             self.up2 = CoupledUpReal(256, [128,64])
#             self.up3 = CoupledUpReal(128, [64,32])
#             self.finalblock = nn.Sequential(
#                     nn.Conv2d(64, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
#                     nn.ReLU(),
#                     nn.BatchNorm2d(32),
#                     nn.Conv2d(32, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
#                     nn.ReLU(),
#                     nn.BatchNorm2d(32),
#                     nn.Conv2d(32, 1, (3,3), stride = (1,1), padding = (1,1)),
#                 )
#         else:
#             self.down1 = CoupledDown(1, [32,32])
#             self.down2 = CoupledDown(32, [64,64])
#             self.down3 = CoupledDown(64, [128,128])
#             self.up1 = CoupledUp(128, [256,128])
#             self.up2 = CoupledUp(256, [128,64])
#             self.up3 = CoupledUp(128, [64,32])
#             self.finalblock = nn.Sequential(
#                     cmplx_conv.ComplexConv2d(64, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
#                     cmplx_activation.CReLU(),
#                     radial_bn.RadialBatchNorm2d(32),
#                     cmplx_conv.ComplexConv2d(32, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
#                     cmplx_activation.CReLU(),
#                     radial_bn.RadialBatchNorm2d(32),
#                     cmplx_conv.ComplexConv2d(32, 1,     (3,3), stride = (1,1), padding = (1,1)),
#                 )
#         self.train_mode = True

#     def train_mode_set(self, bool = True):
#         self.train_mode = bool
#         self.down1.train_mode_set(bool)
#         self.down2.train_mode_set(bool)
#         self.down3.train_mode_set(bool)
#         self.up1.train_mode_set(bool)
#         self.up2.train_mode_set(bool)
#         self.up3.train_mode_set(bool)

#     def forward(self, x):
#         with torch.no_grad():
#             x1 = (x**2).sum(1).unsqueeze(1)

#         x2hat, x2 = self.down1(x1)
#         x3hat, x3 = self.down2(x2)
#         x4hat, x4 = self.down3(x3)
#         x5 = self.up1(x4)
#         x6 = self.up2(torch.cat((x5,x4hat),1))
#         x7 = self.up3(torch.cat((x6,x3hat),1))
#         if self.train_mode:
#             x8 = self.finalblock(torch.cat((x7,x2hat),1))
#         else:
#             x8 = no_bn_forward(self.finalblock, torch.cat((x7,x2hat),1))
#         return x8


class ImageSpaceModel1(nn.Module):
    def __init__(self, parameters, proc_device):
        super(ImageSpaceModel1, self).__init__()
        self.param_dic = parameters
        self.image_space_real = self.param_dic['image_space_real']
        self.num_coils = self.param_dic['num_coils']
        if self.param_dic['kspace_combine_coils']:
            self.input_size = 1
        else:
            self.input_size = self.param_dic['num_coils']
        if self.image_space_real:
            self.block1 = nn.Sequential(
                    nn.Conv2d(self.input_size, 2*self.input_size, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm2d(2*self.input_size),
                    nn.Conv2d(2*self.input_size, self.num_coils, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm2d(self.num_coils),
                )
            self.down1 = CoupledDownReal(self.num_coils, [32,32])
            self.down2 = CoupledDownReal(32, [64,64])
            self.down3 = CoupledDownReal(64, [128,128])
            self.up1 = CoupledUpReal(128, [256,128])
            self.up2 = CoupledUpReal(256, [128,64])
            self.up3 = CoupledUpReal(128, [64,32])
            self.finalblock = nn.Sequential(
                    nn.Conv2d(64, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 1, (3,3), stride = (1,1), padding = (1,1)),
                    nn.ReLU()
                )
        # else:
            # self.block1 = nn.Sequential(
            #         cmplx_conv.ComplexConvd(self.num_coils, 2*self.num_coils, (3,3), stride = (1,1), padding = (1,1), bias = False),
            #         cmplx_activation.CReLU(),
            #         radial_bn.RadialBatchNormd(2*self.num_coils),
            #         cmplx_conv.ComplexConvd(2*self.num_coils, self.num_coils, (3,3), stride = (1,1), padding = (1,1), bias = False),
            #         cmplx_activation.CReLU(),
            #         radial_bn.RadialBatchNormd(self.num_coils)
            #     )
            # self.down1 = CoupledDown(self.num_coils, [32,32])
            # self.down2 = CoupledDown(32, [64,64])
            # self.down3 = CoupledDown(64, [128,128])
            # self.up1 = CoupledUp(128, [256,128])
            # self.up2 = CoupledUp(256, [128,64])
            # self.up3 = CoupledUp(128, [64,32])
            # self.finalblock = nn.Sequential(
            #         cmplx_conv.ComplexConv2d(64, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
            #         cmplx_activation.CReLU(),
            #         radial_bn.RadialBatchNorm2d(32),
            #         cmplx_conv.ComplexConv2d(32, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
            #         cmplx_activation.CReLU(),
            #         radial_bn.RadialBatchNorm2d(32),
            #         cmplx_conv.ComplexConv2d(32, 1,     (3,3), stride = (1,1), padding = (1,1)),
            #     )
        self.train_mode = True

    def train_mode_set(self, bool = True):
        self.train_mode = bool
        self.down1.train_mode_set(bool)
        self.down2.train_mode_set(bool)
        self.down3.train_mode_set(bool)
        self.up1.train_mode_set(bool)
        self.up2.train_mode_set(bool)
        self.up3.train_mode_set(bool)

    def time_analysis(self, x, device):
        start = time.time()
        with torch.no_grad():
            if self.train_mode:
                x1 = self.block1(x)
                # x1 = self.block1(x).view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
            else:
                x1 = no_bn_forward(self.block1, x)
                # x1 = no_bn_forward(self.block1, x).view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
            x2hat, x2 = self.down1(x1)
            x3hat, x3 = self.down2(x2)
            x4hat, x4 = self.down3(x3)
            x5 = self.up1(x4)
            x6 = self.up2(torch.cat((x5,x4hat),1))
            x7 = self.up3(torch.cat((x6,x3hat),1))
            if self.train_mode:
                x8 = self.finalblock(torch.cat((x7,x2hat),1))
            else:
                x8 = no_bn_forward(self.finalblock, torch.cat((x7,x2hat),1))
        return time.time() - start

    def forward(self, x):
        if self.train_mode:
            x1 = self.block1(x)
            # x1 = self.block1(x).view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
        else:
            x1 = no_bn_forward(self.block1, x)
            # x1 = no_bn_forward(self.block1, x).view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
        x2hat, x2 = self.down1(x1)
        x3hat, x3 = self.down2(x2)
        x4hat, x4 = self.down3(x3)
        x5 = self.up1(x4)
        x6 = self.up2(torch.cat((x5,x4hat),1))
        x7 = self.up3(torch.cat((x6,x3hat),1))
        if self.train_mode:
            x8 = self.finalblock(torch.cat((x7,x2hat),1))
        else:
            x8 = no_bn_forward(self.finalblock, torch.cat((x7,x2hat),1))
        return x8