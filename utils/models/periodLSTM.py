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


class convLSTMcell(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, tanh_mode = False, sigmoid_mode = True, real_mode = False, theta = False, linear_post_process = False):
        super(convLSTMcell, self).__init__()
        self.tanh_mode = tanh_mode
        self.sigmoid_mode = sigmoid_mode
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.linear_post_process = linear_post_process
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
        if self.linear_post_process:
            if self.real_mode:
                self.linear = nn.Linear(8*8,8*8)
            else:
                self.linear = nn.Linear(8*8*2,8*8*2)

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
        if self.linear_post_process:
            batch_size, coils, res1, res2 = Cthat.shape[:4]
            mid1 = res1//2
            mid2 = res2//2
            inp = Cthat[:,:,mid1-4:mid1+4,mid2-4:mid2+4]
            in_shape = inp.shape
            inp = inp.reshape(batch_size*coils,-1)
            inp = self.linear(inp).reshape(*in_shape)
            Cthat[:,:,mid1-4:mid1+4,mid2-4:mid2+4] = inp
        Ct_new = (ft * prev_state) + (it * Cthat)
        ht = self.activation(Ct_new)*ot

        return Ct_new, ht

class convLSTM_Kspace1(nn.Module):
    def __init__(self, parameters, proc_device, sigmoid_mode = True, two_cell = False):
        super(convLSTM_Kspace1, self).__init__()
        self.param_dic = parameters
        self.history_length = self.param_dic['history_length']
        self.n_coils = self.param_dic['num_coils']

        assert(not(self.param_dic['kspace_coilwise'] and self.param_dic['kspace_coil_combination']))
        assert(not(self.param_dic['ground_truth_enforce'] and self.param_dic['kspace_coil_combination']))

        # if self.param_dic['scale_input_fft']:
        #     tanh_mode = True
        if self.param_dic['kspace_coilwise']:
            in_channels = (1 + self.history_length)
            out_channels = 1
        elif self.param_dic['kspace_coil_combination']:
            in_channels = (1 + self.history_length)*self.n_coils
            out_channels = 1
        else:
            in_channels = (1 + self.history_length)*self.n_coils
            out_channels = self.n_coils

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

        self.mag_m = convLSTMcell(tanh_mode = False, sigmoid_mode = sigmoid_mode, real_mode = True, in_channels = in_channels, out_channels = out_channels)
        self.phase_m = convLSTMcell(
                    tanh_mode = self.param_dic['kspace_tanh'], 
                    sigmoid_mode = sigmoid_mode, 
                    real_mode = self.real_mode, 
                    in_channels = in_channels,
                    out_channels = out_channels,
                    theta = theta, 
                    linear_post_process = self.param_dic['kspace_linear'],
                )
        self.SSIM = kornia.metrics.SSIM(11)
        assert(sigmoid_mode)

    def time_analysis(self, fft_exp, device, periods, ispace_model):
        start = time.time()
        with torch.no_grad():
            mag_log = (fft_exp.abs()+EPS).log()
            phase = fft_exp / (mag_log.exp())
            phase = torch.stack((phase.real, phase.imag), -1)

            prev_state1 = None
            prev_output1 = None
            prev_state2 = None
            prev_output2 = None

            predr = torch.zeros(mag_log.shape)
 
            for ti in range(mag_log.shape[1]):
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

                if self.param_dic['kspace_predict_mode'] == 'thetas':
                    hist_phase = torch.atan2(hist_phase[:,:,:,:,1],hist_phase[:,:,:,:,0])
                elif self.param_dic['kspace_predict_mode'] == 'cosine':
                    hist_phase = hist_phase[:,:,:,:,0]
                elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                    hist_phase = hist_phase
                else:
                    assert 0

                prev_state1, prev_output1 = self.phase_m(hist_phase, prev_state1, prev_output1)
                prev_state2, prev_output2 = self.mag_m(hist_mag, prev_state2, prev_output2)

                if self.param_dic['kspace_predict_mode'] == 'cosine':
                    phase_ti = torch.complex(prev_output1, ((1-(prev_output1**2)) + EPS)**0.5)
                elif self.param_dic['kspace_predict_mode'] == 'thetas':
                    phase_ti = torch.complex(torch.cos(prev_output1), torch.sin(prev_output1))
                elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                    prev_output1 = prev_output1 / (((prev_output1**2).sum(-1)+EPS)**0.5).unsqueeze(-1).detach()
                    phase_ti = torch.complex(prev_output1[:,:,:,:,0], prev_output1[:,:,:,:,1])

                predr_ti = torch.fft.ifft2(torch.fft.ifftshift(prev_output2.exp()*phase_ti, dim = (-2,-1))).real.clip(-200,200)
                predr[:,ti,:,:] = ispace_model(predr_ti)
                
                prev_output2 = prev_output2.detach()
                prev_output1 = prev_output1.detach()            

        tot_time = time.time()-start
        return tot_time


    def forward(self, fft_exp, gt_masks, device, periods, targ_phase = None, targ_mag_log = None, targ_real = None, og_video = None):
        # if self.param_dic['kspace_coil_combination'] and (targ_phase is not None):
        #     targ_real = og_video.unsqueeze(2)
        #     fft_targ = torch.fft.fftshift(torch.fft.fft2(og_video.unsqueeze(2)), dim = (-2,-1))
        #     targ_mag_log = (fft_targ.abs()+EPS).log()
        #     targ_phase = fft_targ / targ_mag_log.exp()
        #     targ_phase = torch.stack((targ_phase.real, targ_phase.imag),-1)

        mag_log = (fft_exp.abs()+EPS).log()
        phase = fft_exp / (mag_log.exp())
        del fft_exp

        if self.param_dic['scale_input_fft']:
            bottom_range = (mag_log.min(-1)[0]).min(-1)[0]
            bottom_range = bottom_range.unsqueeze(-1).unsqueeze(-1)
            mag_log = mag_log - bottom_range
            
            range_span = (mag_log.max(-1)[0]).max(-1)[0]
            range_span = range_span.unsqueeze(-1).unsqueeze(-1)
            mag_log = mag_log / range_span

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
        gt_masks = (gt_masks == 1).repeat(1,1,mag_log.shape[2],1,1).cpu()
        centre = self.param_dic['image_resolution']//2
        width = self.param_dic['image_resolution']//8
        gt_masks[:,:,:,:centre-width,:] = False
        gt_masks[:,:,:,centre+width:,:] = False
        gt_masks[:,:,:,:,:centre-width] = False
        gt_masks[:,:,:,:,centre+width:] = False

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
            hist_ind = (torch.arange(self.history_length+1).repeat(mag_log.shape[0],1) - self.history_length)
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

            mult = (torch.arange(mag_log.shape[0])*mag_log.shape[1]).reshape(-1,1)
            hist_ind = hist_ind + mult      



            hist_phase = phase.reshape(-1, *phase.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *phase.shape[3:])
            hist_mag = mag_log.reshape(-1, *mag_log.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *mag_log.shape[3:])

            if self.param_dic['kspace_predict_mode'] == 'thetas':
                hist_phase = torch.atan2(hist_phase[:,:,:,:,1],hist_phase[:,:,:,:,0])
            elif self.param_dic['kspace_predict_mode'] == 'cosine':
                hist_phase = hist_phase[:,:,:,:,0]
            elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                hist_phase = hist_phase
            else:
                assert 0

            if self.param_dic['kspace_coilwise']:
                if self.real_mode:
                    B, n_coil, r, c = hist_phase.shape
                else:
                    B, n_coil, r, c, _ = hist_phase.shape

                if self.real_mode:    
                    hist_phase = hist_phase.reshape(B*n_coil,1,r,c)
                else:
                    hist_phase = hist_phase.reshape(B*n_coil,1,r,c,2)
                hist_mag = hist_mag.reshape(B*n_coil,1,r,c)
                if prev_state1 is not None:
                    if self.real_mode:    
                        prev_output1 = prev_output1.reshape(B*n_coil,1,r,c)
                        prev_state1 = prev_state1.reshape(B*n_coil,1,r,c)
                    else:
                        prev_output1 = prev_output1.reshape(B*n_coil,1,r,c,2)
                        prev_state1 = prev_state1.reshape(B*n_coil,1,r,c,2)
                    prev_output2 = prev_output2.reshape(B*n_coil,1,r,c)
                    prev_state2 = prev_state2.reshape(B*n_coil,1,r,c)
            
            prev_state1, prev_output1 = self.phase_m(hist_phase, prev_state1, prev_output1)
            prev_state2, prev_output2 = self.mag_m(hist_mag, prev_state2, prev_output2)

            del hist_mag
            del hist_phase
            
            if self.param_dic['kspace_coilwise']:
                if self.real_mode:
                    prev_output1 = prev_output1.reshape(B,n_coil,r,c)
                else:
                    prev_output1 = prev_output1.reshape(B,n_coil,r,c,2)
                prev_output2 = prev_output2.reshape(B,n_coil,r,c)

            if self.param_dic['ground_truth_enforce']:
                if self.param_dic['kspace_predict_mode'] == 'cosine':
                    temp1 = phase[:,ti,:,:,:,0]
                elif self.param_dic['kspace_predict_mode'] == 'thetas':
                    temp1 = torch.atan2(phase[:,ti,:,:,:,1],phase[:,ti,:,:,:,0])
                elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                    temp1 = phase[:,ti,:,:,:]
                else:
                    assert 0
                prev_output1[gt_masks[:,ti,:,:,:]] = temp1[gt_masks[:,ti,:,:,:]]
                temp2 = mag_log[:,ti,:,:,:]
                prev_output2[gt_masks[:,ti,:,:,:]] = temp2[gt_masks[:,ti,:,:,:]]
                del temp1
                del temp2

            if self.param_dic['scale_input_fft']:
                prev_output2 = prev_output2 * (range_span_ti/2)
                prev_output2 = prev_output2 + bottom_range_ti+1

            if self.param_dic['kspace_predict_mode'] == 'cosine':
                phase_ti = torch.complex(prev_output1, ((1-(prev_output1**2)) + EPS)**0.5)
            elif self.param_dic['kspace_predict_mode'] == 'thetas':
                phase_ti = torch.complex(torch.cos(prev_output1), torch.sin(prev_output1))
            elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                prev_output1 = prev_output1 / (((prev_output1**2).sum(-1)+EPS)**0.5).unsqueeze(-1).detach()
                phase_ti = torch.complex(prev_output1[:,:,:,:,0], prev_output1[:,:,:,:,1])

            stacked_phase = torch.stack((phase_ti.real, phase_ti.imag), -1)

            predr_ti = torch.fft.ifft2(torch.fft.ifftshift(prev_output2.exp()*phase_ti, dim = (-2,-1))).real.clip(-200,200)
            del phase_ti
            
            if ti >= self.param_dic['init_skip_frames']:
                if targ_phase is not None:
                    loss_mag += criterionL1(prev_output2*dists, targ_mag_log[:,ti,:,:,:].to(device)*dists)/mag_log.shape[1]
                    if self.param_dic['loss_phase'] == 'L1':
                        loss_phase += criterionL1(stacked_phase, targ_phase[:,ti,:,:,:,:].to(device))/mag_log.shape[1]
                    elif self.param_dic['loss_phase'] == 'Cosine':
                        loss_phase += (1 - criterionCos(stacked_phase, targ_phase[:,ti,:,:,:,:].to(device))).mean()/mag_log.shape[1]
                    elif self.param_dic['loss_phase'] == 'raw_L1':
                        if self.param_dic['kspace_predict_mode'] == 'cosine':
                            loss_phase += criterionL1(prev_output1, targ_phase[:,ti,:,:,:,0].to(device))/mag_log.shape[1]
                        elif self.param_dic['kspace_predict_mode'] == 'thetas':
                            targ_angles = torch.atan2(targ_phase[:,ti,:,:,:,1],targ_phase[:,ti,:,:,:,0]).to(device)
                            loss_phase += criterionL1(prev_output1, targ_angles)/mag_log.shape[1]
                        elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                            loss_phase += criterionL1(stacked_phase, targ_phase[:,ti,:,:,:,:].to(device))/mag_log.shape[1]
                        else:
                            assert 0
                    else:
                        assert 0
                    
                    targ_now = targ_real[:,ti,:,:,:].to(device)
                    loss_real += criterionL1(predr_ti, targ_now)/mag_log.shape[1]
                
                    with torch.no_grad():
                        loss_l1 += (predr_ti- targ_now).reshape(predr_ti.shape[0]*predr_ti.shape[1], predr_ti.shape[2]*predr_ti.shape[3]).abs().mean(1).sum().detach().cpu()
                        loss_l2 += (((predr_ti- targ_now).reshape(predr_ti.shape[0]*predr_ti.shape[1], predr_ti.shape[2]*predr_ti.shape[3]) ** 2).mean(1).sum()).detach().cpu()
                        ss1 = self.SSIM(predr_ti.reshape(predr_ti.shape[0]*predr_ti.shape[1],1,*predr_ti.shape[2:]), targ_now.reshape(predr_ti.shape[0]*predr_ti.shape[1],1,*predr_ti.shape[2:]))
                        ss1 = ss1.reshape(ss1.shape[0],-1)
                        loss_ss1 += ss1.mean(1).sum().detach().cpu()

            prev_output2 = prev_output2.detach()
            prev_output1 = prev_output1.detach()            

            ans_mag_log[:,ti,:,:] = prev_output2.detach()
            ans_phase[:,ti,:,:,:,:] = stacked_phase.detach()
            predr[:,ti,:,:] = predr_ti.detach()

        return predr, ans_phase, ans_mag_log, loss_mag, loss_phase, loss_real, (loss_l1, loss_l2, loss_ss1)

class convLSTM_Kspace2(nn.Module):
    def __init__(self, parameters, proc_device, sigmoid_mode = True, two_cell = False):
        super(convLSTM_Kspace2, self).__init__()
        self.param_dic = parameters
        self.history_length = self.param_dic['history_length']
        self.n_coils = self.param_dic['num_coils']

        assert(not(self.param_dic['kspace_coilwise'] and self.param_dic['kspace_coil_combination']))
        assert(not(self.param_dic['ground_truth_enforce'] and self.param_dic['kspace_coil_combination']))

        # if self.param_dic['scale_input_fft']:
        #     tanh_mode = True
        if self.param_dic['kspace_coilwise']:
            in_channels = (1 + self.history_length)
            out_channels = 1
        elif self.param_dic['kspace_coil_combination']:
            in_channels = (1 + self.history_length)*self.n_coils
            out_channels = 1
        else:
            in_channels = (1 + self.history_length)*self.n_coils
            out_channels = self.n_coils

        theta = False

        self.model = convLSTMcell(tanh_mode = False, sigmoid_mode = sigmoid_mode, real_mode = False, in_channels = in_channels, out_channels = out_channels, theta = False)
        self.SSIM = kornia.metrics.SSIM(11)
        assert(sigmoid_mode)

    def time_analysis(self, fft_exp, device, periods, ispace_model):
        start = time.time()
        with torch.no_grad():
            real = fft_exp
            phase = fft_exp / (mag_log.exp())
            phase = torch.stack((phase.real, phase.imag), -1)

            prev_state1 = None
            prev_output1 = None
            prev_state2 = None
            prev_output2 = None

            predr = torch.zeros(mag_log.shape)
 
            for ti in range(mag_log.shape[1]):
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

                if self.param_dic['kspace_predict_mode'] == 'thetas':
                    hist_phase = torch.atan2(hist_phase[:,:,:,:,1],hist_phase[:,:,:,:,0])
                elif self.param_dic['kspace_predict_mode'] == 'cosine':
                    hist_phase = hist_phase[:,:,:,:,0]
                elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                    hist_phase = hist_phase
                else:
                    assert 0

                prev_state1, prev_output1 = self.phase_m(hist_phase, prev_state1, prev_output1)
                prev_state2, prev_output2 = self.mag_m(hist_mag, prev_state2, prev_output2)

                if self.param_dic['kspace_predict_mode'] == 'cosine':
                    phase_ti = torch.complex(prev_output1, ((1-(prev_output1**2)) + EPS)**0.5)
                elif self.param_dic['kspace_predict_mode'] == 'thetas':
                    phase_ti = torch.complex(torch.cos(prev_output1), torch.sin(prev_output1))
                elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                    prev_output1 = prev_output1 / (((prev_output1**2).sum(-1)+EPS)**0.5).unsqueeze(-1).detach()
                    phase_ti = torch.complex(prev_output1[:,:,:,:,0], prev_output1[:,:,:,:,1])

                predr_ti = torch.fft.ifft2(torch.fft.ifftshift(prev_output2.exp()*phase_ti, dim = (-2,-1))).real.clip(-200,200)
                predr[:,ti,:,:] = ispace_model(predr_ti)
                
                prev_output2 = prev_output2.detach()
                prev_output1 = prev_output1.detach()            

        tot_time = time.time()-start
        return tot_time


    def forward(self, fft_exp, gt_masks, device, periods, targ_phase = None, targ_mag_log = None, targ_real = None, og_video = None):
        # if self.param_dic['kspace_coil_combination'] and (targ_phase is not None):
        #     targ_real = og_video.unsqueeze(2)
        #     fft_targ = torch.fft.fftshift(torch.fft.fft2(og_video.unsqueeze(2)), dim = (-2,-1))
        #     targ_mag_log = (fft_targ.abs()+EPS).log()
        #     targ_phase = fft_targ / targ_mag_log.exp()
        #     targ_phase = torch.stack((targ_phase.real, targ_phase.imag),-1)

        inp = (fft_exp.abs()+CEPS).log()
        inp = torch.stack((inp.real,inp.imag), -1)
        predr = torch.zeros(fft_exp.shape)
        del fft_exp

        prev_state1 = None
        prev_output1 = None


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

        for ti in range(inp.shape[1]):
            hist_ind = (torch.arange(self.history_length+1).repeat(inp.shape[0],1) - self.history_length)
            hist_ind = hist_ind * periods.reshape(-1,1).cpu()
            hist_ind += ti
            temp1 = hist_ind.clone()
            temp1[temp1 < 0] = 9999999999
            min_vals = temp1.min(1)[0]
            base = torch.zeros(temp1.shape) + min_vals.reshape(-1,1)

            hist_ind = hist_ind - base
            hist_ind[hist_ind < 0] = 0
            hist_ind = (hist_ind + base).long()

            mult = (torch.arange(inp.shape[0])*inp.shape[1]).reshape(-1,1)
            hist_ind = hist_ind + mult
            hist_inp = inp.reshape(-1, *inp.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *inp.shape[3:])

            prev_state1, prev_output1 = self.model(hist_inp, prev_state1, prev_output1)
            
            predr_ti = torch.fft.ifft2(torch.fft.ifftshift(torch.complex(prev_output1[:,:,:,:,0], prev_output1[:,:,:,:,1]), dim = (-2,-1))).real.clip(-200,200)
            
            if ti >= self.param_dic['init_skip_frames']:
                if targ_phase is not None:
                    # loss_mag += criterionL1(prev_output2*dists, targ_mag_log[:,ti,:,:,:].to(device)*dists)/mag_log.shape[1]
                    # if self.param_dic['loss_phase'] == 'L1':
                    #     loss_phase += criterionL1(stacked_phase, targ_phase[:,ti,:,:,:,:].to(device))/mag_log.shape[1]
                    # elif self.param_dic['loss_phase'] == 'Cosine':
                    #     loss_phase += (1 - criterionCos(stacked_phase, targ_phase[:,ti,:,:,:,:].to(device))).mean()/mag_log.shape[1]
                    # elif self.param_dic['loss_phase'] == 'raw_L1':
                    #     if self.param_dic['kspace_predict_mode'] == 'cosine':
                    #         loss_phase += criterionL1(prev_output1, targ_phase[:,ti,:,:,:,0].to(device))/mag_log.shape[1]
                    #     elif self.param_dic['kspace_predict_mode'] == 'thetas':
                    #         targ_angles = torch.atan2(targ_phase[:,ti,:,:,:,1],targ_phase[:,ti,:,:,:,0]).to(device)
                    #         loss_phase += criterionL1(prev_output1, targ_angles)/mag_log.shape[1]
                    #     elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                    #         loss_phase += criterionL1(stacked_phase, targ_phase[:,ti,:,:,:,:].to(device))/mag_log.shape[1]
                    #     else:
                    #         assert 0
                    # else:
                    #     assert 0
                    
                    targ_now = targ_real[:,ti,:,:,:].to(device)
                    loss_real += criterionL1(predr_ti, targ_now)/inp.shape[1]
                
                    with torch.no_grad():
                        loss_l1 += (predr_ti- targ_now).reshape(predr_ti.shape[0]*predr_ti.shape[1], predr_ti.shape[2]*predr_ti.shape[3]).abs().mean(1).sum().detach().cpu()
                        loss_l2 += (((predr_ti- targ_now).reshape(predr_ti.shape[0]*predr_ti.shape[1], predr_ti.shape[2]*predr_ti.shape[3]) ** 2).mean(1).sum()).detach().cpu()
                        ss1 = self.SSIM(predr_ti.reshape(predr_ti.shape[0]*predr_ti.shape[1],1,*predr_ti.shape[2:]), targ_now.reshape(predr_ti.shape[0]*predr_ti.shape[1],1,*predr_ti.shape[2:]))
                        ss1 = ss1.reshape(ss1.shape[0],-1)
                        loss_ss1 += ss1.mean(1).sum().detach().cpu()

            # prev_output2 = prev_output2.detach()
            prev_output1 = prev_output1.detach()            

            # ans_mag_log[:,ti,:,:] = prev_output2.detach()
            # ans_phase[:,ti,:,:,:,:] = stacked_phase.detach()
            predr[:,ti,:,:] = predr_ti.detach()

        # return predr, ans_phase, ans_mag_log, loss_mag, loss_phase, loss_real, (loss_l1, loss_l2, loss_ss1)
        if loss_real is not None:
            return predr, None, None, loss_real.detach()*0, loss_real.detach()*0, loss_real, (loss_l1, loss_l2, loss_ss1)
        else:
            return predr, None, None, loss_real, loss_real, loss_real, (loss_l1, loss_l2, loss_ss1)


class convLSTM_Ispace1(nn.Module):
    def __init__(self, parameters, proc_device, sigmoid_mode = True, two_cell = False):
        super(convLSTM_Ispace1, self).__init__()
        self.param_dic = parameters
        self.history_length = self.param_dic['history_length']
        self.n_coils = self.param_dic['num_coils']

        assert(not(self.param_dic['kspace_coilwise'] and self.param_dic['kspace_coil_combination']))
        assert(not(self.param_dic['ground_truth_enforce'] and self.param_dic['kspace_coil_combination']))

        # if self.param_dic['scale_input_fft']:
        #     tanh_mode = True
        if self.param_dic['kspace_coilwise']:
            in_channels = (1 + self.history_length)
            out_channels = 1
        elif self.param_dic['kspace_coil_combination']:
            in_channels = (1 + self.history_length)*self.n_coils
            out_channels = 1
        else:
            in_channels = (1 + self.history_length)*self.n_coils
            out_channels = self.n_coils

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

        self.mag_m = convLSTMcell(tanh_mode = False, sigmoid_mode = sigmoid_mode, real_mode = True, in_channels = in_channels, out_channels = out_channels)
        self.phase_m = convLSTMcell(
                    tanh_mode = self.param_dic['kspace_tanh'], 
                    sigmoid_mode = sigmoid_mode, 
                    real_mode = self.real_mode, 
                    in_channels = in_channels,
                    out_channels = out_channels,
                    theta = theta, 
                    linear_post_process = self.param_dic['kspace_linear'],
                )
        self.SSIM = kornia.metrics.SSIM(11)
        assert(sigmoid_mode)

    def time_analysis(self, fft_exp, device, periods, ispace_model):
        start = time.time()
        with torch.no_grad():
            mag_log = (fft_exp.abs()+EPS).log()
            phase = fft_exp / (mag_log.exp())
            phase = torch.stack((phase.real, phase.imag), -1)

            prev_state1 = None
            prev_output1 = None
            prev_state2 = None
            prev_output2 = None

            predr = torch.zeros(mag_log.shape)
 
            for ti in range(mag_log.shape[1]):
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

                if self.param_dic['kspace_predict_mode'] == 'thetas':
                    hist_phase = torch.atan2(hist_phase[:,:,:,:,1],hist_phase[:,:,:,:,0])
                elif self.param_dic['kspace_predict_mode'] == 'cosine':
                    hist_phase = hist_phase[:,:,:,:,0]
                elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                    hist_phase = hist_phase
                else:
                    assert 0

                prev_state1, prev_output1 = self.phase_m(hist_phase, prev_state1, prev_output1)
                prev_state2, prev_output2 = self.mag_m(hist_mag, prev_state2, prev_output2)

                if self.param_dic['kspace_predict_mode'] == 'cosine':
                    phase_ti = torch.complex(prev_output1, ((1-(prev_output1**2)) + EPS)**0.5)
                elif self.param_dic['kspace_predict_mode'] == 'thetas':
                    phase_ti = torch.complex(torch.cos(prev_output1), torch.sin(prev_output1))
                elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                    prev_output1 = prev_output1 / (((prev_output1**2).sum(-1)+EPS)**0.5).unsqueeze(-1).detach()
                    phase_ti = torch.complex(prev_output1[:,:,:,:,0], prev_output1[:,:,:,:,1])

                predr_ti = torch.fft.ifft2(torch.fft.ifftshift(prev_output2.exp()*phase_ti, dim = (-2,-1))).real.clip(-200,200)
                predr[:,ti,:,:] = ispace_model(predr_ti)
                
                prev_output2 = prev_output2.detach()
                prev_output1 = prev_output1.detach()            

        tot_time = time.time()-start
        return tot_time


    def forward(self, fft_exp, gt_masks, device, periods, targ_phase = None, targ_mag_log = None, targ_real = None, og_video = None):
        # if self.param_dic['kspace_coil_combination'] and (targ_phase is not None):
        #     targ_real = og_video.unsqueeze(2)
        #     fft_targ = torch.fft.fftshift(torch.fft.fft2(og_video.unsqueeze(2)), dim = (-2,-1))
        #     targ_mag_log = (fft_targ.abs()+EPS).log()
        #     targ_phase = fft_targ / targ_mag_log.exp()
        #     targ_phase = torch.stack((targ_phase.real, targ_phase.imag),-1)

        mag_log = (fft_exp.abs()+EPS).log()
        phase = fft_exp / (mag_log.exp())
        del fft_exp

        if self.param_dic['scale_input_fft']:
            bottom_range = (mag_log.min(-1)[0]).min(-1)[0]
            bottom_range = bottom_range.unsqueeze(-1).unsqueeze(-1)
            mag_log = mag_log - bottom_range
            
            range_span = (mag_log.max(-1)[0]).max(-1)[0]
            range_span = range_span.unsqueeze(-1).unsqueeze(-1)
            mag_log = mag_log / range_span

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
        gt_masks = (gt_masks == 1).repeat(1,1,mag_log.shape[2],1,1).cpu()
        centre = self.param_dic['image_resolution']//2
        width = self.param_dic['image_resolution']//8
        gt_masks[:,:,:,:centre-width,:] = False
        gt_masks[:,:,:,centre+width:,:] = False
        gt_masks[:,:,:,:,:centre-width] = False
        gt_masks[:,:,:,:,centre+width:] = False

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
            hist_ind = (torch.arange(self.history_length+1).repeat(mag_log.shape[0],1) - self.history_length)
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

            mult = (torch.arange(mag_log.shape[0])*mag_log.shape[1]).reshape(-1,1)
            hist_ind = hist_ind + mult      



            hist_phase = phase.reshape(-1, *phase.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *phase.shape[3:])
            hist_mag = mag_log.reshape(-1, *mag_log.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *mag_log.shape[3:])

            if self.param_dic['kspace_predict_mode'] == 'thetas':
                hist_phase = torch.atan2(hist_phase[:,:,:,:,1],hist_phase[:,:,:,:,0])
            elif self.param_dic['kspace_predict_mode'] == 'cosine':
                hist_phase = hist_phase[:,:,:,:,0]
            elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                hist_phase = hist_phase
            else:
                assert 0

            if self.param_dic['kspace_coilwise']:
                if self.real_mode:
                    B, n_coil, r, c = hist_phase.shape
                else:
                    B, n_coil, r, c, _ = hist_phase.shape

                if self.real_mode:    
                    hist_phase = hist_phase.reshape(B*n_coil,1,r,c)
                else:
                    hist_phase = hist_phase.reshape(B*n_coil,1,r,c,2)
                hist_mag = hist_mag.reshape(B*n_coil,1,r,c)
                if prev_state1 is not None:
                    if self.real_mode:    
                        prev_output1 = prev_output1.reshape(B*n_coil,1,r,c)
                        prev_state1 = prev_state1.reshape(B*n_coil,1,r,c)
                    else:
                        prev_output1 = prev_output1.reshape(B*n_coil,1,r,c,2)
                        prev_state1 = prev_state1.reshape(B*n_coil,1,r,c,2)
                    prev_output2 = prev_output2.reshape(B*n_coil,1,r,c)
                    prev_state2 = prev_state2.reshape(B*n_coil,1,r,c)
            
            prev_state1, prev_output1 = self.phase_m(hist_phase, prev_state1, prev_output1)
            prev_state2, prev_output2 = self.mag_m(hist_mag, prev_state2, prev_output2)

            del hist_mag
            del hist_phase
            
            if self.param_dic['kspace_coilwise']:
                if self.real_mode:
                    prev_output1 = prev_output1.reshape(B,n_coil,r,c)
                else:
                    prev_output1 = prev_output1.reshape(B,n_coil,r,c,2)
                prev_output2 = prev_output2.reshape(B,n_coil,r,c)

            if self.param_dic['ground_truth_enforce']:
                if self.param_dic['kspace_predict_mode'] == 'cosine':
                    temp1 = phase[:,ti,:,:,:,0]
                elif self.param_dic['kspace_predict_mode'] == 'thetas':
                    temp1 = torch.atan2(phase[:,ti,:,:,:,1],phase[:,ti,:,:,:,0])
                elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                    temp1 = phase[:,ti,:,:,:]
                else:
                    assert 0
                prev_output1[gt_masks[:,ti,:,:,:]] = temp1[gt_masks[:,ti,:,:,:]]
                temp2 = mag_log[:,ti,:,:,:]
                prev_output2[gt_masks[:,ti,:,:,:]] = temp2[gt_masks[:,ti,:,:,:]]
                del temp1
                del temp2

            if self.param_dic['scale_input_fft']:
                prev_output2 = prev_output2 * (range_span_ti/2)
                prev_output2 = prev_output2 + bottom_range_ti+1

            if self.param_dic['kspace_predict_mode'] == 'cosine':
                phase_ti = torch.complex(prev_output1, ((1-(prev_output1**2)) + EPS)**0.5)
            elif self.param_dic['kspace_predict_mode'] == 'thetas':
                phase_ti = torch.complex(torch.cos(prev_output1), torch.sin(prev_output1))
            elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                prev_output1 = prev_output1 / (((prev_output1**2).sum(-1)+EPS)**0.5).unsqueeze(-1).detach()
                phase_ti = torch.complex(prev_output1[:,:,:,:,0], prev_output1[:,:,:,:,1])

            stacked_phase = torch.stack((phase_ti.real, phase_ti.imag), -1)

            predr_ti = torch.fft.ifft2(torch.fft.ifftshift(prev_output2.exp()*phase_ti, dim = (-2,-1))).real.clip(-200,200)
            del phase_ti
            
            if ti >= self.param_dic['init_skip_frames']:
                if targ_phase is not None:
                    loss_mag += criterionL1(prev_output2*dists, targ_mag_log[:,ti,:,:,:].to(device)*dists)/mag_log.shape[1]
                    if self.param_dic['loss_phase'] == 'L1':
                        loss_phase += criterionL1(stacked_phase, targ_phase[:,ti,:,:,:,:].to(device))/mag_log.shape[1]
                    elif self.param_dic['loss_phase'] == 'Cosine':
                        loss_phase += (1 - criterionCos(stacked_phase, targ_phase[:,ti,:,:,:,:].to(device))).mean()/mag_log.shape[1]
                    elif self.param_dic['loss_phase'] == 'raw_L1':
                        if self.param_dic['kspace_predict_mode'] == 'cosine':
                            loss_phase += criterionL1(prev_output1, targ_phase[:,ti,:,:,:,0].to(device))/mag_log.shape[1]
                        elif self.param_dic['kspace_predict_mode'] == 'thetas':
                            targ_angles = torch.atan2(targ_phase[:,ti,:,:,:,1],targ_phase[:,ti,:,:,:,0]).to(device)
                            loss_phase += criterionL1(prev_output1, targ_angles)/mag_log.shape[1]
                        elif self.param_dic['kspace_predict_mode'] == 'unit-vector':
                            loss_phase += criterionL1(stacked_phase, targ_phase[:,ti,:,:,:,:].to(device))/mag_log.shape[1]
                        else:
                            assert 0
                    else:
                        assert 0
                    
                    targ_now = targ_real[:,ti,:,:,:].to(device)
                    loss_real += criterionL1(predr_ti, targ_now)/mag_log.shape[1]
                
                    with torch.no_grad():
                        loss_l1 += (predr_ti- targ_now).reshape(predr_ti.shape[0]*predr_ti.shape[1], predr_ti.shape[2]*predr_ti.shape[3]).abs().mean(1).sum().detach().cpu()
                        loss_l2 += (((predr_ti- targ_now).reshape(predr_ti.shape[0]*predr_ti.shape[1], predr_ti.shape[2]*predr_ti.shape[3]) ** 2).mean(1).sum()).detach().cpu()
                        ss1 = self.SSIM(predr_ti.reshape(predr_ti.shape[0]*predr_ti.shape[1],1,*predr_ti.shape[2:]), targ_now.reshape(predr_ti.shape[0]*predr_ti.shape[1],1,*predr_ti.shape[2:]))
                        ss1 = ss1.reshape(ss1.shape[0],-1)
                        loss_ss1 += ss1.mean(1).sum().detach().cpu()

            prev_output2 = prev_output2.detach()
            prev_output1 = prev_output1.detach()            

            ans_mag_log[:,ti,:,:] = prev_output2.detach()
            ans_phase[:,ti,:,:,:,:] = stacked_phase.detach()
            predr[:,ti,:,:] = predr_ti.detach()

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
        if self.param_dic['kspace_coil_combination']:
            self.input_size = 1
        else:
            self.input_size = self.num_coils
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