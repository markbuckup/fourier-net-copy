import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms
import numpy as np
import os
import sys
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

    if kspace_name == 'KLSTM1':
        km = convLSTM_Kspace1
    if kspace_name == 'KLSTM2':
        km = convLSTM_Kspace2
    return km, im


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
                cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.out_channels, self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                # relu_func(),
            )
        self.forgetGate = nn.Sequential(
                cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.out_channels, self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                # relu_func(),
            )
        self.outputGate = nn.Sequential(
                cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.out_channels, self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                # relu_func(),
            )
        self.inputProc = nn.Sequential(
                cnn_func(self.out_channels + self.in_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.out_channels, 2*self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.out_channels, self.out_channels, (3,3), stride = (1,1), padding = (1,1)),
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

class convLSTM_Kspace1(nn.Module):
    def __init__(self, parameters, proc_device, tanh_mode = False, sigmoid_mode = True, two_cell = False):
        super(convLSTM_Kspace1, self).__init__()
        self.param_dic = parameters
        self.history_length = self.param_dic['history_length']
        self.n_coils = self.param_dic['num_coils']
        # if self.param_dic['scale_input_fft']:
        #     tanh_mode = True
        self.phase_m = convLSTMcell(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode, real_mode = False, in_channels = (1 + self.history_length)*self.n_coils, out_channels = self.n_coils)
        self.mag_m = convLSTMcell(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode, real_mode = True, in_channels = (1 + self.history_length)*self.n_coils, out_channels = self.n_coils)
        assert(not tanh_mode)
        self.SSIM = kornia.metrics.SSIM(11)
        assert(sigmoid_mode)

    def forward(self, fft_exp, undersample_mask, device, periods, targ_phase = None, targ_mag_log = None, targ_real = None):
        mag_log = (fft_exp.abs()+EPS).log()
        phase = fft_exp / (mag_log.exp())

        if self.param_dic['scale_input_fft']:
            bottom_range = (mag_log.min(-1)[0]).min(-1)[0]
            bottom_range = bottom_range.unsqueeze(-1).unsqueeze(-1)
            mag_log = mag_log - bottom_range
            
            range_span = (mag_log.max(-1)[0]).max(-1)[0]
            range_span = range_span.unsqueeze(-1).unsqueeze(-1)
            mag_log = mag_log / range_span

        if undersample_mask is not None:
            mag_log = mag_log * undersample_mask
            phase = phase * undersample_mask
        phase = torch.stack((phase.real, phase.imag), -1)

        prev_state1 = None
        prev_output1 = None
        prev_state2 = None
        prev_output2 = None

        ans_mag_log = torch.zeros(mag_log.shape)
        length = ans_mag_log.shape[-1]
        x_size, y_size = length, length
        x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
        cell = (length//2, length//2)
        dists = ((x_arr - cell[0])**2 + (y_arr - cell[1])**2)**0.33
        dists = torch.FloatTensor(dists).to(device)
        ans_phase = torch.zeros(phase.shape)
        predr = torch.zeros(mag_log.shape)
        
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

        for ti in range(fft_exp.shape[1]):
            hist_ind = (torch.arange(self.history_length+1).repeat(fft_exp.shape[0],1) - self.history_length)
            hist_ind = hist_ind * periods.reshape(-1,1).cpu()
            hist_ind += ti
            temp1 = hist_ind.clone()
            temp1[temp1 < 0] = 9999999999
            min_vals = temp1.min(1)[0]
            base = torch.zeros(temp1.shape) + min_vals.reshape(-1,1)

            if self.param_dic['scale_input_fft']:
                bottom_range_ti = targ_mag_log[:,ti:ti+1].min(-1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1)
                range_span_ti = (targ_mag_log[:,ti:ti+1]-bottom_range_ti).max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1)

            hist_ind = hist_ind - base
            hist_ind[hist_ind < 0] = 0
            hist_ind = (hist_ind + base).long()

            mult = (torch.arange(fft_exp.shape[0])*fft_exp.shape[1]).reshape(-1,1)
            hist_ind = hist_ind + mult            

            hist_phase = phase.reshape(-1, *phase.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *phase.shape[3:])
            hist_mag = mag_log.reshape(-1, *mag_log.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *mag_log.shape[3:])

            prev_state1, prev_output1 = self.phase_m(hist_phase.to(device), prev_state1, prev_output1)
            prev_state2, prev_output2 = self.mag_m(hist_mag.to(device), prev_state2, prev_output2)
            if self.param_dic['scale_input_fft']:
                prev_output2 = prev_output2 * (range_span_ti/2)
                prev_output2 = prev_output2 + bottom_range_ti+1

            phase_ti = torch.complex(prev_output1[:,:,:,:,0], prev_output1[:,:,:,:,1])
            pred_ft = prev_output2.exp()*phase_ti
            # predr_ti = special_trim(torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).real)
            
            # predr_ti = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).real.clip(-5,5)
            # predr_ti = predr_ti - predr_ti.min().detach()
            # predr_ti = predr_ti / (EPS + predr_ti.max().detach())
            predr_ti = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).real.clip(-200,200)
            # predr_ti = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).abs().clip(-200,200)
            # scalar_min = predr_ti.min(-1)[0].min(-1)[0].unsqueeze(2).unsqueeze(3).detach()
            # scalar_min.requires_grad = False
            # predr_ti = predr_ti - scalar_min
            # scalar_max = predr_ti.max(-1)[0].max(-1)[0].unsqueeze(2).unsqueeze(3).detach()
            # scalar_max.requires_grad = False
            # predr_ti = predr_ti / (1e-4 + scalar_max)
            
            mag_temp = ((ans_phase**2).sum(-1)**0.5).unsqueeze(-1)
            ans_phase = ans_phase / (mag_temp.detach() + EPS)


            if ti >= self.param_dic['init_skip_frames']:
                if targ_phase is not None:
                    loss_mag += criterionL1(prev_output2*dists, targ_mag_log[:,ti,:,:,:].to(device)*dists)/fft_exp.shape[1]
                    loss_phase += (1 - criterionCos(prev_output1, targ_phase[:,ti,:,:,:,:].to(device))).mean()/fft_exp.shape[1]
                    
                    targ_now = targ_real[:,ti,:,:,:].to(device)
                    loss_real += criterionL1(predr_ti, targ_now)/fft_exp.shape[1]
                
                    with torch.no_grad():
                        loss_l1 += (predr_ti- targ_now).reshape(predr_ti.shape[0]*predr_ti.shape[1], predr_ti.shape[2]*predr_ti.shape[3]).abs().mean(1).sum().detach().cpu()
                        loss_l2 += (((predr_ti- targ_now).reshape(predr_ti.shape[0]*predr_ti.shape[1], predr_ti.shape[2]*predr_ti.shape[3]) ** 2).mean(1).sum()).detach().cpu()
                        ss1 = self.SSIM(predr_ti.reshape(predr_ti.shape[0]*predr_ti.shape[1],1,*predr_ti.shape[2:]), targ_now.reshape(predr_ti.shape[0]*predr_ti.shape[1],1,*predr_ti.shape[2:]))
                        ss1 = ss1.reshape(ss1.shape[0],-1)
                        loss_ss1 += ss1.mean(1).sum().detach().cpu()

            prev_output2 = prev_output2.detach()
            prev_output1 = prev_output1.detach()            

            ans_mag_log[:,ti,:,:] = prev_output2.cpu()
            ans_phase[:,ti,:,:,:] = prev_output1.cpu()
            predr[:,ti,:,:] = predr_ti.cpu().detach()

        return predr, ans_phase.cpu(), ans_mag_log.cpu(), loss_mag, loss_phase, loss_real, (loss_l1, loss_l2, loss_ss1)

class convLSTM_Kspace2(nn.Module):
    def __init__(self, parameters, proc_device, tanh_mode = False, sigmoid_mode = True, two_cell = False):
        super(convLSTM_Kspace2, self).__init__()
        self.param_dic = parameters
        self.history_length = self.param_dic['history_length']
        self.n_coils = self.param_dic['num_coils']
        # if self.param_dic['scale_input_fft']:
        #     tanh_mode = True
        self.phase_m = convLSTMcell(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode, real_mode = False, in_channels = (1 + self.history_length), out_channels = 1)
        self.mag_m = convLSTMcell(tanh_mode = tanh_mode, sigmoid_mode = sigmoid_mode, real_mode = True, in_channels = (1 + self.history_length), out_channels = 1)
        assert(not tanh_mode)
        self.SSIM = kornia.metrics.SSIM(11)
        assert(sigmoid_mode)

    def forward(self, fft_exp, undersample_mask, device, periods, targ_phase = None, targ_mag_log = None, targ_real = None):
        mag_log = (fft_exp.abs()+EPS).log()
        phase = fft_exp / (mag_log.exp())

        if self.param_dic['scale_input_fft']:
            bottom_range = (mag_log.min(-1)[0]).min(-1)[0]
            bottom_range = bottom_range.unsqueeze(-1).unsqueeze(-1)
            mag_log = mag_log - bottom_range
            
            range_span = (mag_log.max(-1)[0]).max(-1)[0]
            range_span = range_span.unsqueeze(-1).unsqueeze(-1)
            mag_log = mag_log / range_span

        if undersample_mask is not None:
            mag_log = mag_log * undersample_mask
            phase = phase * undersample_mask
        phase = torch.stack((phase.real, phase.imag), -1)

        prev_state1 = None
        prev_output1 = None
        prev_state2 = None
        prev_output2 = None

        ans_mag_log = torch.zeros(mag_log.shape)
        length = ans_mag_log.shape[-1]
        x_size, y_size = length, length
        x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
        cell = (length//2, length//2)
        dists = ((x_arr - cell[0])**2 + (y_arr - cell[1])**2)**0.33
        dists = torch.FloatTensor(dists).to(device)
        ans_phase = torch.zeros(phase.shape)
        predr = torch.zeros(mag_log.shape)
        
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

        for ti in range(fft_exp.shape[1]):
            hist_ind = (torch.arange(self.history_length+1).repeat(fft_exp.shape[0],1) - self.history_length)
            hist_ind = hist_ind * periods.reshape(-1,1).cpu()
            hist_ind += ti
            temp1 = hist_ind.clone()
            temp1[temp1 < 0] = 9999999999
            min_vals = temp1.min(1)[0]
            base = torch.zeros(temp1.shape) + min_vals.reshape(-1,1)

            if self.param_dic['scale_input_fft']:
                bottom_range_ti = targ_mag_log[:,ti:ti+1].min(-1)[0].min(-1)[0].unsqueeze(-1).unsqueeze(-1)
                range_span_ti = (targ_mag_log[:,ti:ti+1]-bottom_range_ti).max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1)

            hist_ind = hist_ind - base
            hist_ind[hist_ind < 0] = 0
            hist_ind = (hist_ind + base).long()

            mult = (torch.arange(fft_exp.shape[0])*fft_exp.shape[1]).reshape(-1,1)
            hist_ind = hist_ind + mult            

            hist_phase = phase.reshape(-1, *phase.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *phase.shape[3:])
            hist_mag = mag_log.reshape(-1, *mag_log.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *mag_log.shape[3:])

            B, n_coil, r, c, _ = hist_phase.shape
            hist_phase = hist_phase.reshape(B*n_coil,1,r,c,2)
            hist_mag = hist_mag.reshape(B*n_coil,1,r,c)
            if prev_state1 is not None:
                prev_output1 = prev_output1.reshape(B*n_coil,1,r,c,2)
                prev_output2 = prev_output2.reshape(B*n_coil,1,r,c)
                prev_state1 = prev_state1.reshape(B*n_coil,1,r,c,2)
                prev_state2 = prev_state2.reshape(B*n_coil,1,r,c)

            prev_state1, prev_output1 = self.phase_m(hist_phase.to(device), prev_state1, prev_output1)
            prev_state2, prev_output2 = self.mag_m(hist_mag.to(device), prev_state2, prev_output2)

            prev_output1 = prev_output1.reshape(B,n_coil,r,c,2)
            prev_output2 = prev_output2.reshape(B,n_coil,r,c)

            if self.param_dic['scale_input_fft']:
                prev_output2 = prev_output2 * (range_span_ti/2)
                prev_output2 = prev_output2 + bottom_range_ti+1

            phase_ti = torch.complex(prev_output1[:,:,:,:,0], prev_output1[:,:,:,:,1])
            pred_ft = prev_output2.exp()*phase_ti
            # predr_ti = special_trim(torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).real)
            
            # predr_ti = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).real.clip(-5,5)
            # predr_ti = predr_ti - predr_ti.min().detach()
            # predr_ti = predr_ti / (EPS + predr_ti.max().detach())
            predr_ti = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).real.clip(-200,200)
            # predr_ti = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).abs().clip(-200,200)
            # scalar_min = predr_ti.min(-1)[0].min(-1)[0].unsqueeze(2).unsqueeze(3).detach()
            # scalar_min.requires_grad = False
            # predr_ti = predr_ti - scalar_min
            # scalar_max = predr_ti.max(-1)[0].max(-1)[0].unsqueeze(2).unsqueeze(3).detach()
            # scalar_max.requires_grad = False
            # predr_ti = predr_ti / (1e-4 + scalar_max)
            
            mag_temp = ((ans_phase**2).sum(-1)**0.5).unsqueeze(-1)
            ans_phase = ans_phase / (mag_temp.detach() + EPS)


            if ti >= self.param_dic['init_skip_frames']:
                if targ_phase is not None:
                    loss_mag += criterionL1(prev_output2*dists, targ_mag_log[:,ti,:,:,:].to(device)*dists)/fft_exp.shape[1]
                    loss_phase += (1 - criterionCos(prev_output1, targ_phase[:,ti,:,:,:,:].to(device))).mean()/fft_exp.shape[1]
                    
                    targ_now = targ_real[:,ti,:,:,:].to(device)
                    loss_real += criterionL1(predr_ti, targ_now)/fft_exp.shape[1]
                
                    with torch.no_grad():
                        loss_l1 += (predr_ti- targ_now).reshape(predr_ti.shape[0]*predr_ti.shape[1], predr_ti.shape[2]*predr_ti.shape[3]).abs().mean(1).sum().detach().cpu()
                        loss_l2 += (((predr_ti- targ_now).reshape(predr_ti.shape[0]*predr_ti.shape[1], predr_ti.shape[2]*predr_ti.shape[3]) ** 2).mean(1).sum()).detach().cpu()
                        ss1 = self.SSIM(predr_ti.reshape(predr_ti.shape[0]*predr_ti.shape[1],1,*predr_ti.shape[2:]), targ_now.reshape(predr_ti.shape[0]*predr_ti.shape[1],1,*predr_ti.shape[2:]))
                        ss1 = ss1.reshape(ss1.shape[0],-1)
                        loss_ss1 += ss1.mean(1).sum().detach().cpu()

            prev_output2 = prev_output2.detach()
            prev_output1 = prev_output1.detach()            

            ans_mag_log[:,ti,:,:] = prev_output2.cpu()
            ans_phase[:,ti,:,:,:] = prev_output1.cpu()
            predr[:,ti,:,:] = predr_ti.cpu().detach()

        return predr, ans_phase.cpu(), ans_mag_log.cpu(), loss_mag, loss_phase, loss_real, (loss_l1, loss_l2, loss_ss1)

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

class ImageSpaceModel2(nn.Module):
    def __init__(self, parameters, proc_device):
        super(ImageSpaceModel2, self).__init__()
        self.param_dic = parameters
        self.image_space_real = self.param_dic['image_space_real']
        self.num_coils = self.param_dic['num_coils']
        if self.image_space_real:
            self.down1 = CoupledDownReal(1, [32,32])
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
                )
        else:
            self.down1 = CoupledDown(1, [32,32])
            self.down2 = CoupledDown(32, [64,64])
            self.down3 = CoupledDown(64, [128,128])
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
        with torch.no_grad():
            x1 = (x**2).sum(1).unsqueeze(1)

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


class ImageSpaceModel1(nn.Module):
    def __init__(self, parameters, proc_device):
        super(ImageSpaceModel1, self).__init__()
        self.param_dic = parameters
        self.image_space_real = self.param_dic['image_space_real']
        self.num_coils = self.param_dic['num_coils']
        if self.image_space_real:
            self.block1 = nn.Sequential(
                    nn.Conv2d(self.num_coils, 2*self.num_coils, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm2d(2*self.num_coils),
                    nn.Conv2d(2*self.num_coils, self.num_coils, (3,3), stride = (1,1), padding = (1,1), bias = False),
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
                )
        else:
            self.block1 = nn.Sequential(
                    cmplx_conv.ComplexConvd(self.num_coils, 2*self.num_coils, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNormd(2*self.num_coils),
                    cmplx_conv.ComplexConvd(2*self.num_coils, self.num_coils, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNormd(self.num_coils)
                )
            self.down1 = CoupledDown(self.num_coils, [32,32])
            self.down2 = CoupledDown(32, [64,64])
            self.down3 = CoupledDown(64, [128,128])
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