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
from utils.models.Recon import Recon1 as Model

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action = 'store_true')
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--test_only', action = 'store_true')
parser.add_argument('--visualise_only', action = 'store_true')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--port', type = int, default = 12355)
parser.add_argument('--state', type = int, default = -1)
parser.add_argument('--gpu', nargs='+', type = int, default = [-1])
parser.add_argument('--neptune_log', action = 'store_true')
# parser.add_argument('--gpu', type = int, default = '-1')
args = parser.parse_args()

LR = 3e-4
EPOCHS = 100
parameters = {}
parameters['train_batch_size'] = 23
parameters['test_batch_size'] = 23
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

model = Model().to(device)
optimiser = optim.Adam(list(model.parameters()), lr = LR)
criterion = nn.L1Loss().to(device)

def train(epoch):
    loss_av = 0
    coil_mask = get_golden_bars(num_bars = 14, resolution = 64)
    coil_mask = coil_mask.sum(0).sign().unsqueeze(0).unsqueeze(0)
    for i, (indices, data) in tqdm(enumerate(trainloader), total = len(trainloader), desc = "[{}] | Epoch {}".format(os.getpid(), epoch), position=0, leave = True):
        optimiser.zero_grad(set_to_none = True)
        batch, num_frames, chan, numr, numc = data.shape
        data = data.reshape(batch*num_frames, chan, numr, numc)
        inpt = (torch.fft.fftshift(torch.fft.fft2(data.to(device)), dim = (-2,-1))+1e-10)

        model.train()
        pred_ft = model(inpt)

        loss = criterion(pred_ft, inpt)
        # predr = torch.fft.ifft2(torch.fft.ifftshift(pred, dim = (-2,-1))).real
        # predi = torch.fft.ifft2(torch.fft.ifftshift(pred, dim = (-2,-1))).real
        # pred = (predi**2 + predr**2 + 1e-8)**0.5
        
        # loss = criterion(pred, data.to(device))
        
        loss.backward()
        optimiser.step()

        loss_av += loss.item()/len(trainloader)

    print('Average loss for epoch {} = {}'.format(epoch, loss_av), flush = True)
    return loss_av

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
        for i, (indices, data) in tqdm(enumerate(dloader), total = len(dloader), desc = "Testing for Epoch {}".format(epoch), position=0, leave = True):
            batch, num_frames, chan, numr, numc = data.shape
            data = data.reshape(batch*num_frames, chan, numr, numc)
            inpt = (torch.fft.fftshift(torch.fft.fft2(data.to(device)), dim = (-2,-1))+1e-10)
            
            model.eval()

            pred_ft = model(inpt)

            predr = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).real
            target = torch.fft.ifft2(torch.fft.ifftshift(inpt, dim = (-2,-1))).real
            
            for i in range(min(5,pred_ft.shape[0])):
                fig = plt.figure(figsize = (8,8))
                plt.subplot(2,2,1)
                plt.imshow(inpt.log()[i,0,:,:].abs().cpu().numpy(), cmap = 'gray')
                plt.title('Input FFT')
                plt.subplot(2,2,2)
                plt.imshow(pred_ft.log()[i,0,:,:].abs().cpu().numpy(), cmap = 'gray')
                plt.title('Predicted FFT')
                plt.subplot(2,2,3)
                plt.imshow(target.cpu()[i,0,:,:], cmap = 'gray')
                plt.title('Actual Image')
                plt.subplot(2,2,4)
                plt.imshow(predr[i,0,:,:].cpu().numpy(), cmap = 'gray')
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