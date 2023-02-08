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

class convLSTMcell(nn.Module):
    def __init__(self, num_channels = 1, tanh_mode = False, sigmoid_mode = True):
        super(convLSTMcell, self).__init__()
        self.tanh_mode = tanh_mode
        self.sigmoid_mode = sigmoid_mode
        self.num_channels = num_channels
        self.inputGate = nn.Sequential(
                cmplx_conv.ComplexConv2d(2*self.num_channels, 2*self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(2*self.num_channels, 2*self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(2*self.num_channels, self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                # cmplx_activation.CReLU(),
            )
        self.forgetGate = nn.Sequential(
                cmplx_conv.ComplexConv2d(2*self.num_channels, 2*self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(2*self.num_channels, 2*self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(2*self.num_channels, self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                # cmplx_activation.CReLU(),
            )
        self.outputGate = nn.Sequential(
                cmplx_conv.ComplexConv2d(2*self.num_channels, 2*self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(2*self.num_channels, 2*self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(2*self.num_channels, self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                # cmplx_activation.CReLU(),
            )
        self.inputProc = nn.Sequential(
                cmplx_conv.ComplexConv2d(2*self.num_channels, 2*self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(2*self.num_channels, 2*self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(2*self.num_channels, self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                # cmplx_activation.CReLU(),
            )
        # self.decoder = nn.Linear(64*64,64*64)
        # self.decoder2 = nn.Linear(64*64*2, 64*64)

    def forward(self, x, prev_state = None, prev_output = None):
        # x is a batch of video frames at a single time stamp
        if prev_state is None:
            shape1 = (x.shape[0], 1, *x.shape[2:])
            shape2 = (x.shape[0], 64*64)
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
            ht = Ct_new*ot
        # ht = self.decoder2(torch.cat((ht_temp, x.reshape(-1,64*64)),1)).view(-1,1,64,64)

        return Ct_new, ht

class convLSTM(nn.Module):
    def __init__(self, tanh_mode = False, sigmoid_mode = True, two_cell = False):
        super(convLSTM, self).__init__()
        self.cell1 = convLSTMcell(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode)
        self.two_cell = two_cell
        if self.two_cell:
            self.cell2 = convLSTMcell(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode)

    def forward(self, fft_exp, coil_mask, device):
        fft_log = (fft_exp+1e-8).log()*coil_mask
        fft_log = torch.stack((fft_log.real, fft_log.imag), -1)
        # x => B, t, C, h,w
        prev_state1 = None
        prev_output1 = None
        if self.two_cell:
            prev_state2 = None
            prev_output2 = None
        ans_log = fft_log.clone().detach()
        for ti in range(fft_log.shape[1]):
            prev_state1, prev_output1 = self.cell1(fft_log[:,ti,:,:,:].to(device), prev_state1, prev_output1)
            if self.two_cell:
                prev_state2, prev_output2 = self.cell2(prev_output1, prev_state2, prev_output2)
                ans_log[:,ti,:,:,:] = prev_output2
            else:
                ans_log[:,ti,:,:,:] = prev_output1
        ans_log = torch.complex(ans_log[:,:,:,:,:,0], ans_log[:,:,:,:,:,1])
        return ans_log

class convLSTM_quad(nn.Module):
    def __init__(self, tanh_mode = False, sigmoid_mode = True, two_cell = False):
        super(convLSTM_quad, self).__init__()
        self.m1 = convLSTM(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode, two_cell = two_cell)
        self.m2 = convLSTM(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode, two_cell = two_cell)

    def forward(self, fft_exp, coil_mask, device):
        fft_exp2 = torch.complex(fft_exp.real.clip(-2,2), fft_exp.imag.clip(-2,2)).exp()
        
        ans_log1 = self.m1(fft_exp, coil_mask, device)
        ans_2 = (self.m2(fft_exp, coil_mask, device) + 1e-8).log()

        return ans_log1 + ans_2