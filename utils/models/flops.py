from torchvision.io import write_video
from torchvision.io import read_video
import torch
import torchvision
from fvcore.nn import FlopCountAnalysis

from MDCNN import MDCNN
from periodLSTM import convLSTM_Kspace1, ImageSpaceModel1
from params_flops.params_mdcnn import parameters as parameters_mdcnn
from params_flops.params_lstm import parameters as parameters_lstm

def spec_format(num):
    if num > 1000000000:
        if not num % 1000000000:
            return f'{num // 1000000000}B'
        return f'{round(num / 1000000000, 1)}B'
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000}M'
        return f'{round(num / 1000000, 1)}M'
    return f'{num // 1000}K'

mdcnn = MDCNN(parameters_mdcnn)
lstm = convLSTM_Kspace1(parameters_lstm, torch.device('cpu'))
ispace = ImageSpaceModel1(parameters_lstm, torch.device('cpu'))

flops_mdcnn_k = FlopCountAnalysis(mdcnn.kspacem, torch.zeros(1,8,7,256,256,2))
flops_mdcnn_i = FlopCountAnalysis(mdcnn.imspacem, torch.zeros(1,8,7,256,256))
flops_lstm_m = FlopCountAnalysis(lstm.mag_m, torch.zeros(1,8,256,256))
flops_lstm_p = FlopCountAnalysis(lstm.phase_m, torch.zeros(1,8,256,256))
flops_ispace = FlopCountAnalysis(ispace, torch.zeros(1,8,256,256))

print('MDCNN K Space FLOPs = {}'.format(spec_format(flops_mdcnn_k.total())))
print('MDCNN I Space FLOPs = {}'.format(spec_format(flops_mdcnn_i.total())))
print('MDCNN Total FLOPs = {}'.format(spec_format(flops_mdcnn_k.total()+flops_mdcnn_i.total())))

print('LSTM K Space FLOPs = {}'.format(spec_format(flops_lstm_m.total()+flops_lstm_p.total())))
print('LSTM I Space FLOPs = {}'.format(spec_format(flops_ispace.total())))
print('LSTM Total FLOPs = {}'.format(spec_format(flops_lstm_m.total()+flops_lstm_p.total()+flops_ispace.total())))

