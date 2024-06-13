import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms
import numpy as np
import os
import kornia
import sys
import time
sys.path.append('../')
sys.path.append('../../')
import matplotlib.pyplot as plt
import utils.models.complexCNNs.cmplx_conv as cmplx_conv
import utils.models.complexCNNs.cmplx_dropout as cmplx_dropout
import utils.models.complexCNNs.cmplx_upsample as cmplx_upsample
import utils.models.complexCNNs.cmplx_activation as cmplx_activation
import utils.models.complexCNNs.radial_bn as radial_bn

EPS = 1e-10
CEPS = torch.complex(torch.tensor(EPS),torch.tensor(EPS)).exp()

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
            x2 = self.block2(x1) + x1
        else:
            x1 = no_bn_forward(self.block1, x) + x
            x2 = no_bn_forward(self.block2, x1) + x1
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


class ImageSpaceModel(nn.Module):
    def __init__(self, num_coils = 8, num_window = 7, image_space_real = False):
        super(ImageSpaceModel, self).__init__()
        self.image_space_real = image_space_real
        self.num_coils = num_coils
        if self.image_space_real:
            self.block1 = nn.Sequential(
                    nn.Conv3d(self.num_coils, 2*self.num_coils, (3,3,3), stride = (1,1,1), padding = (1,1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm3d(2*self.num_coils),
                    nn.Conv3d(2*self.num_coils, self.num_coils, (3,3,3), stride = (1,1,1), padding = (1,1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm3d(self.num_coils),
                )
            self.down1 = CoupledDownReal(num_coils*num_window, [32,32])
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
                    nn.Conv2d(32, 1, (1,1), stride = (1,1), padding = (0,0)),
                )
        else:
            self.block1 = nn.Sequential(
                    cmplx_conv.ComplexConv3d(self.num_coils, 2*self.num_coils, (3,3,3), stride = (1,1,1), padding = (1,1,1), bias = True),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNorm3d(2*self.num_coils),
                    cmplx_conv.ComplexConv3d(2*self.num_coils, self.num_coils, (3,3,3), stride = (1,1,1), padding = (1,1,1), bias = True),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNorm3d(self.num_coils)
                )
            self.down1 = CoupledDown(num_coils*num_window, [32,32])
            self.down2 = CoupledDown(32, [64,64])
            self.down3 = CoupledDown(64, [128,128])
            self.up1 = CoupledUp(128, [256,128])
            self.up2 = CoupledUp(256, [128,64])
            self.up3 = CoupledUp(128, [64,32])
            self.finalblock = nn.Sequential(
                    cmplx_conv.ComplexConv2d(64, 32, (3,3), stride = (1,1), padding = (1,1), bias = True),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNorm2d(32),
                    cmplx_conv.ComplexConv2d(32, 32, (3,3), stride = (1,1), padding = (1,1), bias = True),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNorm2d(32),
                    cmplx_conv.ComplexConv2d(32, 1, (1,1), stride = (1,1), padding = (0,0)),
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
            x1 = self.block1(x)
            # x1 = self.block1(x).view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
        else:
            x1 = no_bn_forward(self.block1, x)
            # x1 = no_bn_forward(self.block1, x).view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
        if self.image_space_real:
            x1 = x1.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
        else:
            x1 = x1.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4], x.shape[5])
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

class ImageSpaceDecoder(nn.Module):
    def __init__(self, num_coils = 8, num_window = 7, image_space_real = False):
        super(ImageSpaceDecoder, self).__init__()
        self.image_space_real = image_space_real
        self.num_coils = num_coils
        if self.image_space_real:
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
    def forward_decode(self, x4, x2hat, x3hat, x4hat):
        x5 = self.up1(x4)
        x6 = self.up2(torch.cat((x5,x4hat),1))
        x7 = self.up3(torch.cat((x6,x3hat),1))
        x8 = self.finalblock(torch.cat((x7,x2hat),1))
        return x8
        

class ImageSpaceEncoder(nn.Module):
    def __init__(self, num_coils = 8, num_window = 7, image_space_real = False):
        super(ImageSpaceEncoder, self).__init__()
        self.image_space_real = image_space_real
        self.num_coils = num_coils
        if self.image_space_real:
            self.block1 = nn.Sequential(
                    nn.Conv3d(self.num_coils, 2*self.num_coils, (3,3,3), stride = (1,1,1), padding = (1,1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm3d(2*self.num_coils),
                    nn.Conv3d(2*self.num_coils, self.num_coils, (3,3,3), stride = (1,1,1), padding = (1,1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm3d(self.num_coils),
                )
            self.down1 = CoupledDownReal(num_coils*num_window, [32,32])
            self.down2 = CoupledDownReal(32, [64,64])
            self.down3 = CoupledDownReal(64, [128,128])
            
        else:
            self.block1 = nn.Sequential(
                    cmplx_conv.ComplexConv3d(self.num_coils, 2*self.num_coils, (3,3,3), stride = (1,1,1), padding = (1,1,1), bias = False),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNorm3d(2*self.num_coils),
                    cmplx_conv.ComplexConv3d(2*self.num_coils, self.num_coils, (3,3,3), stride = (1,1,1), padding = (1,1,1), bias = False),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNorm3d(self.num_coils)
                )
            self.down1 = CoupledDown(num_coils*num_window, [32,32])
            self.down2 = CoupledDown(32, [64,64])
            self.down3 = CoupledDown(64, [128,128])

    def forward(self, x):
        x1 = self.block1(x)
        if self.image_space_real:
            x1 = x1.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
        else:
            x1 = x1.view(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4], x.shape[5])
        x2hat, x2 = self.down1(x1)
        x3hat, x3 = self.down2(x2)
        x4hat, x4 = self.down3(x3)
        return [x4, x2hat, x3hat, x4hat]

class MDCNN(nn.Module):
    def __init__(self, parameters, proc_device):
        super(MDCNN, self).__init__()
        self.param_dic = parameters
        self.proc_device = proc_device
        self.num_window = parameters['window_size']
        assert(self.num_window == 7)
        self.num_coils = parameters['num_coils']
        self.image_space_real = parameters['image_space_real']
        self.kspace_m = KspaceModel(num_coils = self.num_coils)
        self.ispacem = ImageSpaceModel(num_coils = self.num_coils, num_window = self.num_window, image_space_real = self.image_space_real)
        self.train_mode = True
        self.criterionL1 = nn.L1Loss().to(proc_device)
        self.SSIM = kornia.metrics.SSIM(11)

    def time_analysis(self, fft_exp, device, periods, ispace_model):
        with torch.no_grad():
            times = []
            predr = torch.zeros(*fft_exp.shape[0:2], 1, *fft_exp.shape[3:])

            for ti in range(self.num_window - 1, fft_exp.shape[1]):
                start1 = time.time()
                input = fft_exp[:,ti-self.num_window+1:ti+1]
                fft_log = (input+CEPS).log().to(device)
                fft_log = torch.stack((fft_log.real, fft_log.imag), -1)
                
                x1 = self.kspace_m(torch.swapaxes(fft_log, 1,2))
                
                real, imag = torch.unbind(x1, -1)
                fftshifted = torch.complex(real, imag)
                x2 = torch.fft.ifft2(torch.fft.ifftshift(fftshifted.exp(), dim = (-2, -1)))
                if self.image_space_real:
                    x3 = x2.abs()
                    ans = self.ispacem(x3)
                else:
                    x3 = torch.stack([x2.real, x2.imag], dim=-1)
                    ans = (self.ispacem(x3).pow(2).sum(-1)+EPS).pow(0.5)
                    # assert(0)

                predr[:,ti] = ans.detach().cpu()

                times.append(time.time() - start1)
                
        return times


    def forward(self, fft_exp, gt_masks = None, device = torch.device('cpu'), periods = None, targ_phase = None, targ_mag_log = None, targ_real = None, og_video = None, epoch = np.inf):
        fft_log = (fft_exp+CEPS).log()
        fft_log = torch.stack((fft_log.real, fft_log.imag), -1)
        # FT data - b_num, num_coils, num_windows, 256, 256
        # Returns - kspace_data, image_space_data

        predr = torch.zeros(*fft_exp.shape[0:2], 1, *fft_exp.shape[3:]).to(device)

        zero_tensor = torch.tensor(0.).to(device)
        loss_real = zero_tensor
        loss_l1 = zero_tensor
        loss_l2 = zero_tensor
        loss_ss1 = zero_tensor

        for ti in range(self.num_window - 1, fft_log.shape[1]):

            x1 = self.kspace_m(torch.swapaxes(fft_log[:,ti-self.num_window+1:ti+1], 1,2))
            # print(x1.abs().min(), x1.abs().max())
            real, imag = torch.unbind(x1, -1)
            fftshifted = torch.complex(real, imag)
            x2 = torch.fft.ifft2(torch.fft.ifftshift(fftshifted.exp(), dim = (-2, -1)))
            if self.image_space_real:
                x3 = x2.abs()
                ans = self.ispacem(x3)
            else:
                x3 = torch.stack([x2.real, x2.imag], dim=-1)
                ans = (self.ispacem(x3).pow(2).sum(-1)+EPS).pow(0.5)
                # assert(0)
            # print(ans.cpu().detach()[0,0].min(), ans.cpu().detach()[0,0].max())
            if og_video is not None:
                targ_now = og_video[:,ti,:,:]

                # plt.imsave('targ_now.jpg', targ_now.cpu().detach()[0,0], cmap = 'gray')
                # plt.imsave('ans.jpg', ans.cpu().detach()[0,0], cmap = 'gray')

                loss_real += self.criterionL1(ans, targ_now.to(device))
                loss_l1 += (ans - targ_now).reshape(ans .shape[0]*ans .shape[1], -1).abs().mean(1).sum().detach().cpu()
                loss_l2 += (((ans - targ_now).reshape(ans .shape[0]*ans .shape[1], -1) ** 2).mean(1).sum()).detach().cpu()
                ss1 = self.SSIM(ans .reshape(ans .shape[0]*ans .shape[1],1,self.param_dic['image_resolution'],self.param_dic['image_resolution']), targ_now.reshape(ans .shape[0]*ans .shape[1],1,self.param_dic['image_resolution'],self.param_dic['image_resolution']))
                ss1 = ss1.reshape(ss1.shape[0],-1)
                loss_ss1 += ss1.mean(1).sum().detach().cpu()

            predr[:,ti] = ans.detach()

        # return predr, predr_kspace, loss_mag, loss_phase, loss_real, loss_forget_gate, loss_input_gate, (loss_l1, loss_l2, loss_ss1)
        return predr, predr, zero_tensor, zero_tensor, loss_real, zero_tensor, zero_tensor, (loss_l1, loss_l2, loss_ss1)