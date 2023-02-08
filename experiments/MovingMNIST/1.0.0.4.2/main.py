import os
import gc
import sys
import PIL
import time
import torch
import random
import pickle
import argparse
import numpy as np
import torchvision
import neptune.new as neptune
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from torch import nn, optim
from neptune.new.types import File
from torch.nn import functional as F
from torchvision import transforms, models, datasets
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('/root/Cardiac-MRI-Reconstrucion/')
os.environ['display'] = 'localhost:14.0'

from utils.myDatasets.MovingMNIST import MovingMNIST
from utils.functions import get_coil_mask, get_golden_bars
from utils.models.convLSTM import convLSTM4_2 as Model

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action = 'store_true')
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--test_only', action = 'store_true')
parser.add_argument('--visualise_only', action = 'store_true')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--port', type = int, default = 12355)
parser.add_argument('--state', type = int, default = -1)
parser.add_argument('--gpu', nargs='+', type = int, default = [-1])
parser.add_argument('--neptune_log', action = 'store_true', default = False)
# parser.add_argument('--gpu', type = int, default = '-1')
args = parser.parse_args()
EPS = 1e-10
# torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
# # torch.autograd.set_detect_anomaly(False)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)


EPOCHS = 100
parameters = {}
parameters['lr'] = 3e-4
parameters['descrption'] = 'Signle Cell in LSTM, all gradients detached - no tanh -  yes sigmoid, loss on mag and angle'
parameters['train_batch_size'] = 23
parameters['test_batch_size'] = 2
parameters['train_test_split'] = 0.8
parameters['dataset_path'] = '../../../datasets/moving_mnist'
parameters['normalisation'] = True

checkpoint_path = './checkpoints/'
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
if not os.path.isdir('./images/'):
    os.mkdir('./images/')
if not os.path.isdir('./images/Train'):
    os.mkdir('./images/Train')
if not os.path.isdir('./images/Test'):
    os.mkdir('./images/Test')

def convert_cylindrical_to_polar(real,imag):
    """Convert the cylindrical representation (i.e. real and imaginary) to
        polar representation (i.e. magnitude and phase)

    Parameters
    ----------
    real : torch.Tensor
        The real part of the complex tensor
    imag : torch.Tensor
        The imaginary part of the complex tensor
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The magnitude and phase of the complex tensor
    """
    mag = (real ** 2 + imag ** 2 + EPS) ** (0.5)
    phase = torch.atan2(imag, real+EPS)
    phase[phase.ne(phase)] = 0.0  # remove NANs if any
    return mag, phase



if args.gpu == -1:
    device = torch.cpu()
else:
    device = torch.device('cuda:{}'.format(args.gpu[0]))

trainset = MovingMNIST(
                    parameters['dataset_path'], 
                    train = True, 
                    train_split = parameters['train_test_split'], 
                    norm = parameters['normalisation'],
                    encoding = True,
                )
testset = MovingMNIST(
                    parameters['dataset_path'], 
                    train = False, 
                    train_split = parameters['train_test_split'], 
                    norm = parameters['normalisation'],
                    encoding = True
                )

trainloader = torch.utils.data.DataLoader(
                            trainset, 
                            batch_size=parameters['train_batch_size'], 
                            shuffle = True
                        )
traintestloader = torch.utils.data.DataLoader(
                            trainset, 
                            batch_size=parameters['train_batch_size'], 
                            shuffle = False
                        )
testloader = torch.utils.data.DataLoader(
                            testset, 
                            batch_size=parameters['test_batch_size'],
                            shuffle = False
                        )

if args.neptune_log:
    args.run_id = os.getcwd().split('/')[-1]
    if os.path.isfile(checkpoint_path + 'neptune_run.pth'):
        run_id = torch.load(checkpoint_path + 'neptune_run.pth', map_location = torch.device('cpu'))['run_id']
        run = neptune.init_run(
            project="fcrl/FFT-LSTM-Sanity-Checks",
            with_id=run_id,
            name = args.run_id,
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZDU2NDJjMy1lNzczLTRkZDEtODAwYy01MWFlM2VmN2Q4ZTEifQ==",
        )
    else:
        run = neptune.init_run(
            project="fcrl/FFT-LSTM-Sanity-Checks",
            name = args.run_id,
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZDU2NDJjMy1lNzczLTRkZDEtODAwYy01MWFlM2VmN2Q4ZTEifQ==",
        )
        torch.save({'run_id': run["sys/id"].fetch()}, checkpoint_path + 'neptune_run.pth')
    run["parameters"] = parameters
    if not args.resume:
        if run.exists("train"):
            run["train"].pop()
        if run.exists("test"):
            run["test"].pop()
        if run.exists("visualize"):
            run["visualize"].pop()
else:
    run = None

model = Model().to(device)
optimiser = optim.Adam(list(model.parameters()), lr = parameters['lr'])
criterion = nn.L1Loss().to(device)
cos = torch.nn.CosineSimilarity(dim=2, eps=1e-08)
criterion_cos = lambda x,y : (1-cos(x.reshape(*x.shape[:2], -1), y.reshape(*y.shape[:2], -1))).mean()

def train(epoch):
    loss_av = 0
    loss_av_cos = 0
    coil_mask = get_golden_bars(num_bars = 14, resolution = 64)
    coil_mask = coil_mask.sum(0).sign().unsqueeze(0).unsqueeze(0)
    for i, (indices, data) in tqdm(enumerate(trainloader), total = len(trainloader), desc = "[{}] | Ep {}".format(os.getpid(), epoch), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}"):
        optimiser.zero_grad(set_to_none = True)
        batch, num_frames, chan, numr, numc = data.shape
        inpt = (torch.fft.fftshift(torch.fft.fft2(data.to(device)), dim = (-2,-1))+1e-8)

        model.train()
        pred_ft = model(inpt, device)

        mag, phase = convert_cylindrical_to_polar(pred_ft.real, pred_ft.imag)
        mag_targ, phase_targ = convert_cylindrical_to_polar(inpt.log().real, inpt.log().imag)

        mag_loss = criterion(mag, mag_targ)
        phase_loss = criterion_cos(phase, phase_targ)

        loss = mag_loss + phase_loss
        # loss = criterion(pred_ft, inpt.log())
        # predr = torch.fft.ifft2(torch.fft.ifftshift(pred, dim = (-2,-1))).real
        # predi = torch.fft.ifft2(torch.fft.ifftshift(pred, dim = (-2,-1))).real
        # pred = (predi**2 + predr**2 + 1e-8)**0.5
        
        # loss = criterion(pred, data.to(device))
        
        loss.backward()
        optimiser.step()

        loss_av += mag_loss.item()/len(trainloader)
        loss_av_cos += phase_loss.item()/len(trainloader)

    print('Average mag loss for epoch {} = {}'.format(epoch, loss_av), flush = True)
    print('Average phase loss for epoch {} = {}'.format(epoch, loss_av_cos), flush = True)
    return loss_av+loss_av_cos

def eval(epoch, train = False):
    if train:
        dloader = traintestloader
        dset = trainset
        dstr = 'Train'
    else:
        dloader = testloader
        dset = testset
        dstr = 'Test'
    with torch.no_grad():
        coil_mask = get_golden_bars(num_bars = 14, resolution = 64)
        coil_mask = coil_mask.sum(0).sign().unsqueeze(0).unsqueeze(0)
        for i, (indices, data) in tqdm(enumerate(dloader), total = len(dloader), desc = "Testing for Epoch {}".format(epoch), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}"):
            batch, num_frames, chan, numr, numc = data.shape
            inpt = (torch.fft.fftshift(torch.fft.fft2(data), dim = (-2,-1))+1e-8)
            
            model.eval()

            pred_ft = model(inpt, device)

            predr = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).real
            target = torch.fft.ifft2(torch.fft.ifftshift(inpt, dim = (-2,-1))).real
            
            for i in range(pred_ft.shape[1]):
                fig = plt.figure(figsize = (8,8))
                plt.subplot(2,2,1)
                plt.imshow((inpt+1e-8).log()[0,i,0,:,:].abs().cpu().numpy(), cmap = 'gray')
                plt.title('Input FFT')
                plt.subplot(2,2,2)
                plt.imshow((pred_ft+1e-8).log()[0,i,0,:,:].abs().cpu().numpy(), cmap = 'gray')
                plt.title('Predicted FFT')
                plt.subplot(2,2,3)
                plt.imshow(target.cpu()[0,i,0,:,:], cmap = 'gray')
                plt.title('Actual Image')
                plt.subplot(2,2,4)
                plt.imshow(predr[0,i,0,:,:].cpu().numpy(), cmap = 'gray')
                plt.title('Our Reconstructed Image')
                plt.suptitle("{} data Video 0 Frame {}".format(dstr, i))
                plt.savefig('images/{}/epoch{}_{}'.format(dstr, epoch, i))
                plt.close('all')
            break

if args.resume:
    model_state = torch.load(checkpoint_path + 'state.pth', map_location = device)['state']
    if (not args.state == -1):
        model_state = args.state
    print('Loading checkpoint at model state {}'.format(model_state), flush = True)
    dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state), map_location = device)
    pre_e = dic['e']
    model.load_state_dict(dic['model'])
    optimiser.load_state_dict(dic['optimiser'])
    losses = dic['losses']
    print('Resuming Training after {} epochs'.format(pre_e), flush = True)
else:
    model_state = 0
    pre_e =0
    losses = []
    print('Starting Training', flush = True)

if args.eval:
    eval(pre_e, train = True)
    eval(pre_e, train = False)
    plt.figure()
    plt.title('Train Loss')
    plt.plot(range(len(losses)), [x[0] for x in losses], color = 'b')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('images/train_loss.png')
    # plt.figure()
    # plt.title('Test Loss')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.savefig('images/test_loss.png')
    os._exit(0)

for e in range(EPOCHS):
    if pre_e > 0:
        pre_e -= 1
        continue
    loss = train(e)
    losses.append([loss])
    if args.neptune_log:
        run["train/loss"].log(loss)

    dic = {}
    dic['e'] = e+1
    dic['model'] = model.state_dict()
    dic['optimiser'] = optimiser.state_dict()
    dic['losses'] = losses
    
    torch.save(dic, checkpoint_path + 'checkpoint_{}.pth'.format(model_state))
    torch.save({'state': model_state}, checkpoint_path + 'state.pth')
    # model_state += 1
    print('Saving model after {} Epochs\n'.format(e+1), flush = True)
eval(100, train = True)
eval(100, train = False)