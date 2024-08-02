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
    """
    AERS:
    Format a number into a human-readable string with appropriate suffixes.

    The function converts large numbers into a string with 'K', 'M', or 'B' suffixes, representing
    thousands, millions, or billions, respectively.

    Parameters:
    --------------
    - num (int or float): The number to format.

    Returns:
    -----------
    -str: The formatted string with a suffix.
    
    Examples:
    --------------
    >>> spec_format(1500)
    '1K'
    >>> spec_format(1500000)
    '1.5M'
    >>> spec_format(2500000000)
    '2.5B'
    ===================================================================================================
    """
    if num > 1000000000:
        if not num % 1000000000:
            return f'{num // 1000000000}B'
        return f'{round(num / 1000000000, 1)}B'
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000}M'
        return f'{round(num / 1000000, 1)}M'
    return f'{num // 1000}K'

mdcnn = MDCNN(parameters_mdcnn, torch.device('cpu'))
fouriernet = convLSTM_Kspace1(parameters_lstm, torch.device('cpu'))
unet = ImageSpaceModel1(parameters_lstm, torch.device('cpu'))

flops_mdcnn_k = FlopCountAnalysis(mdcnn.kspace_m, torch.zeros(1,8,7,256,256,2))
flops_mdcnn_i = FlopCountAnalysis(mdcnn.ispacem, torch.zeros(1,8,7,256,256,2))
flops_klstm = FlopCountAnalysis(fouriernet.kspace_m, (torch.zeros(1,8,256,256),torch.zeros(1,8,256,256)))
flops_ilstm = FlopCountAnalysis(fouriernet.ispacem, torch.zeros(8,1,256,256))
flops_unet = FlopCountAnalysis(unet, torch.zeros(1,8,256,256))

print('MDCNN K Space FLOPs = {}'.format(spec_format(flops_mdcnn_k.total())))
print('MDCNN I Space FLOPs = {}'.format(spec_format(flops_mdcnn_i.total())))
print('MDCNN Total FLOPs = {}'.format(spec_format(flops_mdcnn_k.total()+flops_mdcnn_i.total())))

print('KLSTM FLOPs = {}'.format(spec_format(flops_klstm.total())))
print('ILSTM FLOPs = {}'.format(spec_format(flops_ilstm.total())))
print('UNET FLOPs = {}'.format(spec_format(flops_unet.total())))
print('FOURIER-Net Total FLOPs = {}'.format(spec_format(flops_klstm.total()+flops_ilstm.total()+flops_unet.total())))

