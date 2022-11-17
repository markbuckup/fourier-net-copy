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
from utils.models.MDCNN import CoupledUp, CoupledUpReal, CoupledDown, CoupledDownReal

EPS = 1e-10

class GRUKspaceModel(nn.Module):
    def __init__(self, input_coils = 16, output_coils = 16):
        '''
        Input size = (B), Coil, X, Y
        '''
        super(GRUKspaceModel, self).__init__()
        self.input_coils = input_coils
        self.output_coils = output_coils
        self.block1 = nn.Sequential(
                cmplx_conv.ComplexConv2d(self.input_coils, 2*self.input_coils, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(2*self.input_coils, self.input_coils, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(self.input_coils, self.input_coils, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
            )
        self.block2 = nn.Sequential(
                cmplx_conv.ComplexConv2d(self.input_coils, 2*self.input_coils, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(2*self.input_coils, self.input_coils, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
                cmplx_conv.ComplexConv2d(self.input_coils, self.input_coils, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
            )
        self.block3 = nn.Sequential(
                cmplx_conv.ComplexConv2d(self.input_coils, self.output_coils, (5,5), stride = (1,1), padding = (2,2)),
                cmplx_activation.CReLU(),
            )

    def forward(self, x):
        x1 = self.block1(x) + x
        x2 = self.block2(x1) + x1
        return self.block3(x2)

class GRUIspaceModel(nn.Module):
    def __init__(self, in_channels = 8, latent_channels = 128, image_space_real = False):
        '''
        Input size = (B), Coil, X, Y
        '''
        super(GRUIspaceModel, self).__init__()
        self.encoder = GRUImageSpaceEncoder(in_channels = in_channels, out_channels = latent_channels, image_space_real = image_space_real)
        self.decoder = GRUImageSpaceDecoder(in_channels = latent_channels, image_space_real = image_space_real)
        
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(*x1)
        return x2

class IFFT_module(nn.Module):
    def __init__(self, parameters):
        super(IFFT_module, self).__init__()
        self.image_space_real = parameters['image_space_real']
    
    def forward(self,x1):
        real, imag = torch.unbind(x1, -1)
        fftshifted = torch.complex(real, imag)
        x2 = torch.fft.ifft2(torch.fft.ifftshift(fftshifted.exp(), dim = (-2, -1)))
        if self.image_space_real:
            x3 = x2.real
        else:
            x3 = torch.stack([x2.real, x2.imag], dim=-1)
        return x3

class GRUImageSpaceDecoder(nn.Module):
    def __init__(self, in_channels = 128, image_space_real = False):
        super(GRUImageSpaceDecoder, self).__init__()
        self.image_space_real = image_space_real
        self.in_channels = in_channels
        if self.image_space_real:
            self.up1 = CoupledUpReal(self.in_channels, [256,128])
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
                )
        else:
            self.up1 = CoupledUp(128, [256,128])
            self.up2 = CoupledUp(256, [128,64])
            self.up3 = CoupledUp(128, [64,32])
            self.finalblock = nn.Sequential(
                    cmplx_conv.ComplexConv2d(64, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNorm2d(32),
                    cmplx_conv.ComplexConv2d(32, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNorm2d(32),
                    cmplx_conv.ComplexConv2d(32, 1,     (3,3), stride = (1,1), padding = (1,1)),
                )
    def forward(self, x4, x2hat, x3hat, x4hat):
        x5 = self.up1(x4)
        x6 = self.up2(torch.cat((x5,x4hat),1))
        x7 = self.up3(torch.cat((x6,x3hat),1))
        x8 = self.finalblock(torch.cat((x7,x2hat),1))
        return x8
        

class GRUImageSpaceEncoder(nn.Module):
    def __init__(self, in_channels = 8, out_channels = 128, image_space_real = False):
        super(GRUImageSpaceEncoder, self).__init__()
        self.image_space_real = image_space_real
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.image_space_real:
            self.block1 = nn.Sequential(
                    nn.Conv2d(self.in_channels, 2*self.in_channels, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm2d(2*self.in_channels),
                    nn.Conv2d(2*self.in_channels, self.in_channels, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm2d(self.in_channels),
                )
            self.down1 = CoupledDownReal(self.in_channels, [32,32])
            self.down2 = CoupledDownReal(32, [64,64])
            self.down3 = CoupledDownReal(64, [self.out_channels, self.out_channels])
            
        else:
            self.block1 = nn.Sequential(
                    cmplx_conv.ComplexConv2d(self.in_channels, 2*self.in_channels, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNorm2d(2*self.in_channels),
                    cmplx_conv.ComplexConv2d(2*self.in_channels, self.in_channels, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNorm2d(self.in_channels)
                )
            self.down1 = CoupledDown(self.in_channels, [32,32])
            self.down2 = CoupledDown(32, [64,64])
            self.down3 = CoupledDown(64, [self.out_channels, self.out_channels])

        self.latent_channels = self.out_channels, 32, 64, self.out_channels

    def forward(self, x):
        x1 = self.block1(x)
        x2hat, x2 = self.down1(x1)
        x3hat, x3 = self.down2(x2)
        x4hat, x4 = self.down3(x3)
        # return [x4, x2hat, x3hat, x4hat], [x4.shape[1],x2hat.shape[1], x3hat.shape[1], x4hat.shape[1]]
        return [x4, x2hat, x3hat, x4hat]