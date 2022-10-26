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
import utils.complexCNNs.cmplx_conv as cmplx_conv
import utils.complexCNNs.cmplx_dropout as cmplx_dropout
import utils.complexCNNs.cmplx_upsample as cmplx_upsample
import utils.complexCNNs.cmplx_activation as cmplx_activation
import utils.complexCNNs.radial_bn as radial_bn

def no_bn_forward(model, x):
    ans = None
    for layer in model:
        if isinstance(layer, radial_bn.RadialBatchNorm2d):
            continue
        else:
            if ans is None:
                ans = layer(x)
            else:
                ans = layer(ans)
    return ans


class KspaceModel(nn.Module):
    def __init__(self, num_coils = 8):
        super(KspaceModel, self).__init__()
        self.num_coils = num_coils
        self.block1 = nn.Sequential(
                cmplx_conv.ComplexConv3d(self.num_coils, 2*self.num_coils, (3,5,5), stride = (1,1,1), padding = (1,2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv3d(2*self.num_coils, 2*self.num_coils, (3,5,5), stride = (1,1,1), padding = (1,2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv3d(2*self.num_coils, self.num_coils, (3,5,5), stride = (1,1,1), padding = (1,2,2)),
                cmplx_activation.CReLU(),
            )
        self.block2 = nn.Sequential(
                cmplx_conv.ComplexConv3d(self.num_coils, 2*self.num_coils, (3,5,5), stride = (1,1,1), padding = (1,2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv3d(2*self.num_coils, 2*self.num_coils, (3,5,5), stride = (1,1,1), padding = (1,2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv3d(2*self.num_coils, self.num_coils, (3,5,5), stride = (1,1,1), padding = (1,2,2)),
                cmplx_activation.CReLU(),
            )
        self.train_mode = True

    def forward(self, x):
        if self.train_mode:
            x1 = self.block1(x) + x
            x2 = self.block1(x1) + x1
        else:
            x1 = no_bn_forward(self.block1, x) + x
            x2 = no_bn_forward(self.block1, x1) + x1
        return x2

    def train_mode_set(self, bool = True):
        self.train_mode = bool

class CoupledDown(nn.Module):
    def __init__(self, inp_channel, outp_channels):
        super(CoupledDown, self).__init__()
        ls = []
        prev = inp_channel
        for i in outp_channels:
            ls.append(cmplx_conv.ComplexConv2d(prev, i, (3,3), stride = (1,1), padding = (1,1), bias = False))
            ls.append(radial_bn.RadialBatchNorm2d(i))
            ls.append(cmplx_activation.CReLU())
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
            ls.append(radial_bn.RadialBatchNorm2d(i))
            ls.append(cmplx_activation.CReLU())
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


class ImageSpaceModel(nn.Module):
    def __init__(self, num_coils = 8, num_window = 7):
        super(ImageSpaceModel, self).__init__()
        self.num_coils = num_coils
        self.block1 = nn.Sequential(
                cmplx_conv.ComplexConv3d(self.num_coils, 2*self.num_coils, (3,3,3), stride = (1,1,1), padding = (1,1,1), bias = False),
                radial_bn.RadialBatchNorm3d(2*self.num_coils),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv3d(2*self.num_coils, self.num_coils, (3,3,3), stride = (1,1,1), padding = (1,1,1), bias = False),
                radial_bn.RadialBatchNorm3d(self.num_coils),
                cmplx_activation.CReLU(),
            )
        self.down1 = CoupledDown(num_coils*num_window, [32,32])
        self.down2 = CoupledDown(32, [64,64])
        self.down3 = CoupledDown(64, [128,128])
        self.up1 = CoupledUp(128, [256,128])
        self.up2 = CoupledUp(256, [128,64])
        self.up3 = CoupledUp(128, [64,32])
        self.finalblock = nn.Sequential(
                cmplx_conv.ComplexConv2d(64, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
                radial_bn.RadialBatchNorm2d(32),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(32, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
                radial_bn.RadialBatchNorm2d(32),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(32, 1,     (3,3), stride = (1,1), padding = (1,1)),
            )
        self.train_mode = True

    def train_mode_set(self, bool = True):
        self.train_mode = bool
        self.down1.train_mode_set(bool)
        self.down2.train_mode_set(bool)
        self.down3.train_mode_set(bool)
        self.up1.train_mode_set(bool)
        self.up2.train_mode_set(bool)
        self.up3.train_mode_set(bool)


    def forward(self, x):
        if self.train_mode:
            x1 = self.block1(x).view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4], x.shape[5])
        else:
            x1 = no_bn_forward(self.block1, x).view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4], x.shape[5])
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

class MDCNN(nn.Module):
    def __init__(self, num_coils, num_window):
        super(MDCNN, self).__init__()
        self.num_window = num_window
        self.num_coils = num_coils
        self.kspacem = KspaceModel(num_coils = num_coils)
        self.imspacem = ImageSpaceModel(num_coils = num_coils, num_window = num_window)
        self.train_mode = True

    def forward(self, x):
        # FT data - b_num, num_coils, num_windows, 256, 256
        # Returns - kspace_data, image_space_data
        x1 = self.kspacem(x)
        real, imag = torch.unbind(x1, -1)
        fftshifted = torch.complex(real, imag)
        x2 = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(fftshifted, dim = (-2, -1))), dim = (-2, -1))
        x3 = torch.stack([x2.real, x2.imag], dim=-1)
        return x1, self.imspacem(x3).pow(2).sum(-1).pow(0.5)

    def train_mode_set(self, bool = True):
        self.train_mode = bool
        self.kspacem.train_mode_set(bool)
        self.imspacem.train_mode_set(bool)