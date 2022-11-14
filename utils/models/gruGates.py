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
from utils.models.gruComponents import GRUKspaceModel, IFFT_module

class GRUGate_KSpace(nn.Module):
    """
    Generate a convolutional GRU cell
    Input Shape = (B), In_Coil, X, Y
    Output Shape = (B), Out_Coil, X, Y
    """
    def __init__(self, parameters):
        super().__init__()
        self.kspace = GRUKspaceModel(input_coils = 2*parameters['num_coils'], output_coils = parameters['num_coils'])

    def forward(self, x, activation = 'sigmoid'):
        if activation == 'sigmoid':
            act = torch.sigmoid
        else:
            act = torch.tanh

        return [act(self.kspace(x[0]))]

class GRUGate_complex2d(nn.Module):
    """
    Generate a convolutional GRU cell
    Input Shape = (B), In_Chan, X, Y
    Output Shape = (B), Out_Chan, X, Y
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.models = []
        for i,(in_channel, out_channel) in enumerate(zip(in_channels, out_channels)):
            mim = nn.Sequential(
                                cmplx_conv.ComplexConv2d(in_channel, 2*in_channel, (3,3), stride = (1,1), padding = (1,1), bias = False),
                                cmplx_activation.CReLU(),
                                radial_bn.RadialBatchNorm2d(2*in_channel),
                                cmplx_conv.ComplexConv2d(2*in_channel, out_channel, (3,3), stride = (1,1), padding = (1,1), bias = False),
                                cmplx_activation.CReLU(),
                                radial_bn.RadialBatchNorm2d(out_channel),
                            )
            name = 'Model_' + str(i).zfill(2)
            setattr(self, name, mim)
            self.models.append(getattr(self, name))
    def forward(self, x, activation = 'sigmoid'):
        if activation == 'sigmoid':
            act = torch.sigmoid
        else:
            act = torch.tanh
        ans = []
        assert(len(x) == len(self.models))
        for i in range(len(x)):
            ans.append(act(self.models[i](x[i])))
        return ans

class GRUGate_real2d(nn.Module):
    """
    Generate a convolutional GRU cell
    Input Shape = (B), In_Chan, X, Y
    Output Shape = (B), Out_Chan, X, Y
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.models = []
        for i,(in_channel, out_channel) in enumerate(zip(in_channels, out_channels)):
            mim = nn.Sequential(
                                nn.Conv2d(in_channel, 2*in_channel, (3,3), stride = (1,1), padding = (1,1), bias = False),
                                nn.ReLU(),
                                nn.BatchNorm2d(2*in_channel),
                                nn.Conv2d(2*in_channel, out_channel, (3,3), stride = (1,1), padding = (1,1), bias = False),
                                nn.ReLU(),
                                nn.BatchNorm2d(out_channel),
                            )
            name = 'Model_' + str(i).zfill(2)
            setattr(self, name, mim)
            self.models.append(getattr(self, name))
    def forward(self, x, activation = 'sigmoid'):
        if activation == 'sigmoid':
            act = torch.sigmoid
        else:
            act = torch.tanh
        ans = []
        assert(len(x) == len(self.models))
        for i in range(len(x)):
            ans.append(act(self.models[i](x[i])))
        return ans