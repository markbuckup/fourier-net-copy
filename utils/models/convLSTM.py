import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms
import numpy as np
import os
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import utils.models.complexCNNs.cmplx_conv as cmplx_conv
import utils.models.complexCNNs.cmplx_dropout as cmplx_dropout
import utils.models.complexCNNs.cmplx_upsample as cmplx_upsample
import utils.models.complexCNNs.cmplx_activation as cmplx_activation
import utils.models.complexCNNs.radial_bn as radial_bn

EPS = 1e-10

class Identity(nn.Module):
    def __init__(self, a):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class convLSTMcell(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, tanh_mode = False, sigmoid_mode = True, real_mode = False, theta = False):
        super(convLSTMcell, self).__init__()
        self.tanh_mode = tanh_mode
        self.sigmoid_mode = sigmoid_mode
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.theta = theta
        if real_mode:
            cnn_func = nn.Conv2d
            relu_func = nn.ReLU
        else:
            cnn_func = cmplx_conv.ComplexConv2d
            relu_func = cmplx_activation.CReLU

        if theta:
            self.activation = lambda x: np.pi*torch.tanh(x)
        else:
            self.activation = lambda x: x


        self.inputGate = nn.Sequential(
                cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (5,5), stride = (1,1), padding = (2,2)),
                relu_func(),
                cnn_func(2*self.out_channels, 2*self.out_channels, (5,5), stride = (1,1), padding = (2,2)),
                relu_func(),
                cnn_func(2*self.out_channels, self.out_channels, (5,5), stride = (1,1), padding = (2,2)),
                # relu_func(),
            )
        self.forgetGate = nn.Sequential(
                cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (5,5), stride = (1,1), padding = (2,2)),
                relu_func(),
                cnn_func(2*self.out_channels, 2*self.out_channels, (5,5), stride = (1,1), padding = (2,2)),
                relu_func(),
                cnn_func(2*self.out_channels, self.out_channels, (5,5), stride = (1,1), padding = (2,2)),
                # relu_func(),
            )
        self.outputGate = nn.Sequential(
                cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (5,5), stride = (1,1), padding = (2,2)),
                relu_func(),
                cnn_func(2*self.out_channels, 2*self.out_channels, (5,5), stride = (1,1), padding = (2,2)),
                relu_func(),
                cnn_func(2*self.out_channels, self.out_channels, (5,5), stride = (1,1), padding = (2,2)),
                # relu_func(),
            )
        self.inputProc = nn.Sequential(
                cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (5,5), stride = (1,1), padding = (2,2)),
                relu_func(),
                cnn_func(2*self.out_channels, 2*self.out_channels, (5,5), stride = (1,1), padding = (2,2)),
                relu_func(),
                cnn_func(2*self.out_channels, self.out_channels, (5,5), stride = (1,1), padding = (2,2)),
                # relu_func(),
            )
        # self.decoder = nn.Linear(64*64,64*64)
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
        if self.tanh_mode: 
            Cthat = torch.tanh(self.inputProc(inp_cat))
            Ct_new = (ft * prev_state) + (it * Cthat)
            ht = torch.tanh(Ct_new)*ot
        else:
            Cthat = self.inputProc(inp_cat)
            Ct_new = (ft * prev_state) + (it * Cthat)
            ht = self.activation(Ct_new*ot)
        # ht = self.decoder2(torch.cat((ht_temp, x.reshape(-1,64*64)),1)).view(-1,1,64,64)

        return Ct_new, ht

class convLSTM(nn.Module):
    def __init__(self, tanh_mode = False, sigmoid_mode = True, two_cell = False):
        super(convLSTM, self).__init__()
        self.phase_m = convLSTMcell(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode, real_mode = False)
        self.mag_m = convLSTMcell(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode, real_mode = True)

    def forward(self, fft_exp, coil_mask, device):
        mag_log = fft_exp.log().real
        mag_exp = mag_log.exp()
        phase = fft_exp / (mag_log.exp())

        mag_log = mag_log * coil_mask
        phase = phase * coil_mask
        phase = torch.stack((phase.real, phase.imag), -1)

        prev_state1 = None
        prev_output1 = None
        prev_state2 = None
        prev_output2 = None

        ans_mag_log = mag_exp.clone().detach()
        ans_phase = phase.clone().detach()

        for ti in range(fft_exp.shape[1]):
            prev_state1, prev_output1 = self.phase_m(phase[:,ti,:,:,:,:].to(device), prev_state1, prev_output1)
            prev_state2, prev_output2 = self.mag_m(mag_log[:,ti,:,:,:].to(device), prev_state2, prev_output2)

            ans_mag_log[:,ti,:,:,:] = prev_output2
            ans_phase[:,ti,:,:,:,:] = prev_output1
            # prev_state1 = prev_state1.detach()
            # prev_state2 = prev_state2.detach()
            prev_output1 = prev_output1.detach()
            prev_output2 = prev_output2.detach()

        mag_temp = ((ans_phase**2).sum(-1)**0.5).unsqueeze(-1)
        ans_phase = ans_phase / (mag_temp.detach() + EPS)

        return ans_phase, ans_mag_log

class convLSTM_quad(nn.Module):
    def __init__(self, tanh_mode = False, sigmoid_mode = True, two_cell = False):
        super(convLSTM_quad, self).__init__()
        self.phase_m = convLSTMcell(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode, real_mode = False)
        self.mag_m1 = convLSTMcell(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode, real_mode = True)
        self.mag_m2 = convLSTMcell(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode, real_mode = True)

    def forward(self, fft_exp, coil_mask, device):
        mag_log = fft_exp.log().real
        mag_exp = mag_log.exp()
        phase = fft_exp / (mag_log.exp())

        mag_log = mag_log * coil_mask
        mag_exp = mag_exp.clip(-2,2) * coil_mask
        phase = phase * coil_mask
        phase = torch.stack((phase.real, phase.imag), -1)

        prev_state1 = None
        prev_output1 = None
        prev_state2 = None
        prev_output2 = None
        prev_state3 = None
        prev_output3 = None

        ans_mag_log = mag_exp.clone().detach()
        ans_phase = phase.clone().detach()

        for ti in range(fft_exp.shape[1]):
            prev_state1, prev_output1 = self.phase_m(phase[:,ti,:,:,:,:].to(device), prev_state1, prev_output1)
            prev_state2, prev_output2 = self.mag_m1(mag_exp[:,ti,:,:,:].to(device), prev_state2, prev_output2)
            prev_state3, prev_output3 = self.mag_m2(mag_log[:,ti,:,:,:].to(device), prev_state3, prev_output3)

            ans_mag_log[:,ti,:,:,:] = prev_output2.log() + prev_output3
            ans_phase[:,ti,:,:,:,:] = prev_output1

        mag_temp = ((ans_phase**2).sum(-1)**0.5).unsqueeze(-1)
        ans_phase = ans_phase / (mag_temp.detach() + EPS)
        
        return ans_phase, ans_mag_log

        
class convLSTM_theta(nn.Module):
    def __init__(self, tanh_mode = False, sigmoid_mode = True, two_cell = False):
        super(convLSTM_theta, self).__init__()
        self.phase_m = convLSTMcell(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode, real_mode = True, theta = True, in_channels = 2, out_channels = 1)
        self.mag_m = convLSTMcell(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode, real_mode = True, in_channels = 1, out_channels = 1)

    def forward(self, fft_exp, coil_mask, device):
        mag_log = fft_exp.log().real
        mag_exp = mag_log.exp()
        phase = fft_exp / (mag_log.exp())

        mag_log = mag_log * coil_mask
        phase = phase * coil_mask
        phase = torch.cat((phase.real, phase.imag), 2)

        prev_state1 = None
        prev_output1 = None
        prev_state2 = None
        prev_output2 = None

        ans_mag_log = mag_exp.clone().detach()
        ans_phase = torch.zeros(phase.shape[0],phase.shape[1],1,phase.shape[3], phase.shape[4], 2, device = fft_exp.device)

        for ti in range(fft_exp.shape[1]):
            prev_state1, prev_output1 = self.phase_m(phase[:,ti,:,:,:].to(device), prev_state1, prev_output1)
            prev_state2, prev_output2 = self.mag_m(mag_log[:,ti,:,:,:].to(device), prev_state2, prev_output2)

            ans_mag_log[:,ti,:,:,:] = prev_output2
            ans_phase[:,ti,:,:,:,0] = torch.cos(prev_output1)
            ans_phase[:,ti,:,:,:,1] = torch.sin(prev_output1)

        # mag_temp = ((ans_phase**2).sum(-1)**0.5).unsqueeze(-1)
        # ans_phase = ans_phase / (mag_temp.detach() + EPS)

        return ans_phase, ans_mag_log

class convLSTM_real(nn.Module):
    def __init__(self, tanh_mode = False, sigmoid_mode = True, two_cell = False):
        super(convLSTM_real, self).__init__()
        self.model = convLSTMcell(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode, real_mode = True)

    def forward(self, x, device):
        prev_state1 = None
        prev_output1 = None
        
        ans = x.clone().detach()
        
        for ti in range(x.shape[1]):
            prev_state1, prev_output1 = self.model(x[:,ti,:,:,:].to(device), prev_state1, prev_output1)
            
            ans[:,ti,:,:,:] = prev_output1

        return ans