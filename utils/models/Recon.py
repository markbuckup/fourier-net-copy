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

class Recon1(nn.Module):
    def __init__(self, num_channels = 1):
        super(Recon1, self).__init__()
        self.num_channels = num_channels
        self.model = nn.Sequential(
                cmplx_conv.ComplexConv2d(self.num_channels, 2*self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(2*self.num_channels, 2*self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(2*self.num_channels, self.num_channels, (5,5), stride = (1,1), padding = (2,2)),
                # cmplx_activation.CReLU(),
            )
        

    def forward(self, orig_fft):
        log_fft = orig_fft.log()
        log_fft = torch.stack((log_fft.real, log_fft.imag), -1)
        ans_log = self.model(log_fft)
        print(ans_log.min(), ans_log.max())
        ans_log = torch.complex(ans_log[:,:,:,:,0], ans_log[:,:,:,:,1])
        return ans_log.exp()