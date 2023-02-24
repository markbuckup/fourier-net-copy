import os
import gc
import sys
import PIL
import time
import torch
import random
import pickle
import kornia
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
from utils.models.convLSTM import convLSTM as Model
from utils.models.convLSTM import convLSTM_real as Model_real
from utils.functions import fetch_loss_function

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

# torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
# # torch.autograd.set_detect_anomaly(False)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)
print(os.getcwd())
EPS = 1e-8
CEPS = torch.complex(torch.tensor(EPS),torch.tensor(EPS))

EPOCHS = 200
parameters = {}
parameters['lr'] = 3e-4
parameters['descrption'] = 'Real Space Trainer'
parameters['train_batch_size'] = 23
parameters['test_batch_size'] = 23
parameters['num_views'] = int(os.getcwd().split('/')[-1].split('.')[-1])
parameters['train_test_split'] = 0.8
parameters['dataset_path'] = '../../../datasets/moving_mnist'
parameters['normalisation'] = True
parameters['loss_params'] = {
    'SSIM_window': 11,
    'alpha_phase': 1,
    'alpha_amp': 1,
    'grayscale': True,
    'deterministic': False,
    'watson_pretrained': True,
}

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

ft_path = '../r.{}/checkpoints/checkpoint_0.pth'.format(int(os.getcwd().split('/')[-1].split('.')[-1]))
model_ft = Model(tanh_mode = False, sigmoid_mode = True).to(device)
dic_ft = torch.load(ft_path, map_location = device)
model_ft.load_state_dict(dic_ft['model'])
for param in model_ft.parameters():
    param.requires_grad = False
model = Model_real(tanh_mode = False, sigmoid_mode = True).to(device)
optimiser = optim.Adam(list(model.parameters()), lr = parameters['lr'])
criterionL1 = nn.L1Loss().to(device)
criterionCos = nn.CosineSimilarity(dim = 5)
SSIM = kornia.metrics.SSIM(11)

def myimshow(x, cmap = 'gray', trim = False):
    if trim:
        percentile_95 = np.percentile(x, 95)
        percentile_5 = np.percentile(x, 5)
        x[x > percentile_95] = percentile_95
        x[x < percentile_5] = percentile_5
    x = x - x.min()
    x = x/ (x.max() + EPS)
    plt.imshow(x, cmap = cmap)

def border_trim(x):
    num_v, num_f, chan, r, c =x.shape
    x = x.reshape(-1, chan*r*c)
    percentile_95 = torch.quantile(x, .95, dim = 1).unsqueeze(1)
    percentile_5 = torch.quantile(x, .5, dim = 1).unsqueeze(1)
    x = x - percentile_5
    x[x<0] = 0
    x = x + percentile_5

    x = x - percentile_95
    x[x>0] = 0
    x = x + percentile_95

    x = x - x.min(1)[0].unsqueeze(1)
    x = x / x.max(1)[0].unsqueeze(1)

    x = x.reshape(num_v, num_f, chan, r, c)
    return x

def train(epoch):
    loss_av = 0
    ssim_score = 0
    coil_mask_coll = get_golden_bars(resolution = 64)
    x_size, y_size = 64,64
    x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
    cell = (32,32)
    dists = ((x_arr - cell[0])**2 + (y_arr - cell[1])**2)**0.33
    dists = torch.FloatTensor(dists).to(device)
    for i, (indices, data) in tqdm(enumerate(trainloader), total = len(trainloader), desc = "[{}] | Training Ep {}".format(os.getpid(), epoch), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}"):
        optimiser.zero_grad(set_to_none = True)
        batch, num_frames, chan, numr, numc = data.shape
        coil_mask = coil_mask_coll[(torch.arange(parameters['num_views']*num_frames)+i)%376,:,:]
        coil_mask = coil_mask.reshape(num_frames,parameters['num_views'],numr,numc).sum(1).sign().unsqueeze(0).unsqueeze(2).to(device)
        inpt = (torch.fft.fftshift(torch.fft.fft2(data.to(device)), dim = (-2,-1))+CEPS)
        
        with torch.no_grad():
            ans_phase, ans_mag_log = model_ft(inpt,coil_mask, device)
            pred_ft = ans_phase*ans_mag_log.unsqueeze(-1).exp()
            pred_ft = torch.complex(pred_ft[:,:,:,:,:,0],pred_ft[:,:,:,:,:,1])
            predr = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).real
            predr = border_trim(predr)[:,21:]
            
        model.train()
        predr_ispace = model(predr.detach(), device)

        loss = criterionL1(predr_ispace, data.to(device)[:,21:])
        
        # loss = criterionL1(pred, data.to(device))

        loss.backward()
        optimiser.step()

        loss_av += loss.item()/len(trainloader)
        ss1 = SSIM(predr_ispace.reshape(-1,chan,numr,numc), data[:,21:].reshape(-1,chan,numr,numc).to(device))
        ss1 = ss1.reshape(ss1.shape[0], -1).mean(1)
        ssim_score += (ss1/(len(trainset)*predr_ispace.shape[1])).sum().item()

    print('Train loss for epoch {} = {}'.format(epoch, loss_av), flush = True)
    print('Train SSIM for epoch {} = {}'.format(epoch, ssim_score), flush = True)
    return loss_av, ssim_score

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
        loss_av = 0
        ssim_score = 0
        coil_mask_coll = get_golden_bars(resolution = 64)
        x_size, y_size = 64,64
        x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
        cell = (32,32)
        dists = ((x_arr - cell[0])**2 + (y_arr - cell[1])**2)**0.33
        dists = torch.FloatTensor(dists).to(device)
        for i, (indices, data) in tqdm(enumerate(dloader), total = len(dloader), desc = "[{}] | Testing Ep {}".format(os.getpid(), epoch), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}"):
            batch, num_frames, chan, numr, numc = data.shape
            coil_mask = coil_mask_coll[(torch.arange(parameters['num_views']*num_frames)+i)%376,:,:]
            coil_mask = coil_mask.reshape(num_frames,parameters['num_views'],numr,numc).sum(1).sign().unsqueeze(0).unsqueeze(2).to(device)
            inpt = (torch.fft.fftshift(torch.fft.fft2(data.to(device)), dim = (-2,-1))+CEPS)
            
            with torch.no_grad():
                ans_phase, ans_mag_log = model_ft(inpt,coil_mask, device)
                pred_ft = ans_phase*ans_mag_log.unsqueeze(-1).exp()
                pred_ft = torch.complex(pred_ft[:,:,:,:,:,0],pred_ft[:,:,:,:,:,1])
                predr = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).real
                predr = border_trim(predr)[:,21:]

            model.eval()
            predr_ispace = model(predr.detach(), device)

            loss = criterionL1(predr_ispace, data.to(device)[:,21:])
            
            # loss = criterionL1(pred, data.to(device))

            loss_av += loss.item()/len(dloader)
            ss1 = SSIM(predr_ispace.reshape(-1,chan,numr,numc), data[:,21:].reshape(-1,chan,numr,numc).to(device))
            ss1 = ss1.reshape(ss1.shape[0], -1).mean(1)
            ssim_score += (ss1/(len(dset)*predr_ispace.shape[1])).sum().item()

    print('Test loss for epoch {} = {}'.format(epoch, loss_av), flush = True)
    print('Test SSIM for epoch {} = {}'.format(epoch, ssim_score), flush = True)
    return loss_av, ssim_score

def visualise(epoch, train = False):
    if train:
        dloader = traintestloader
        dset = trainset
        dstr = 'Train'
    else:
        dloader = testloader
        dset = testset
        dstr = 'Test'
    with torch.no_grad():
        coil_mask_coll = get_golden_bars(resolution = 64)
        for i, (indices, data) in tqdm(enumerate(dloader), total = len(dloader), desc = "Testing for Epoch {}".format(epoch), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}"):
            batch, num_frames, chan, numr, numc = data.shape
            coil_mask = coil_mask_coll[(torch.arange(parameters['num_views']*num_frames)+i)%376,:,:]
            coil_mask = coil_mask.reshape(num_frames,parameters['num_views'],numr,numc).sum(1).sign().unsqueeze(0).unsqueeze(2).to(device)
            inpt = (torch.fft.fftshift(torch.fft.fft2(data.to(device)), dim = (-2,-1))+CEPS)
            

            with torch.no_grad():
                ans_phase, ans_mag_log = model_ft(inpt,coil_mask, device)
                pred_ft = ans_phase*ans_mag_log.unsqueeze(-1).exp()
                pred_ft = torch.complex(pred_ft[:,:,:,:,:,0],pred_ft[:,:,:,:,:,1])
                predr = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).real
                predr = border_trim(predr)[:,21:]


            model.eval()
            predr_ispace = model(predr.detach(), device)
            # predr[:,:,:,:5,:5] = 0
            # predr[:,:,:,-5:,-5:] = 0
            # predr[:,:,:,-5:,:5] = 0
            # predr[:,:,:,:5,-5:] = 0
            target = torch.fft.ifft2(torch.fft.ifftshift(inpt, dim = (-2,-1))).real[:,21:]
            under_targ = torch.fft.ifft2(torch.fft.ifftshift(inpt*coil_mask, dim = (-2,-1))).real[:,21:]
            
            for i in range(target.shape[1]):
                fig = plt.figure(figsize = (12,4))
                plt.subplot(1,3,1)
                myimshow(target.cpu()[0,i,0,:,:], cmap = 'gray')
                plt.title('Image from Complete FFT')
                plt.subplot(1,3,2)
                myimshow(predr[0,i,0,:,:].cpu().numpy(), cmap = 'gray')
                plt.title('Image From KSpace LSTM')
                plt.subplot(1,3,3)
                myimshow(predr_ispace[0,i,0,:,:].cpu().numpy(), cmap = 'gray')
                plt.title('Image From ISpace LSTM')                
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
    test_losses = dic['test_losses']
    print('Resuming Training after {} epochs'.format(pre_e), flush = True)
else:
    model_state = 0
    pre_e =0
    losses = []
    test_losses = []
    print('Starting Training', flush = True)

if args.eval:
    plt.figure()
    plt.plot(range(len(losses)), [x[0] for x in losses], color = 'b', label = 'Train Loss')
    plt.title('Train Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('images/train_loss.png')
    
    plt.figure()
    plt.plot(range(len(losses)), [x[1] for x in losses], color = 'g', label = 'Train SSIM')
    plt.title('Train SSIM')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('images/train_ssim.png')

    plt.figure()
    plt.title('Test Loss')
    plt.plot(range(len(test_losses)), [x[0] for x in test_losses], color = 'b', label = 'Test Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('images/test_loss.png')

    plt.figure()
    plt.title('Test Loss')
    plt.plot(range(len(test_losses)), [x[1] for x in test_losses], color = 'g', label = 'Test SSIM')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('images/test_ssim.png')

    visualise(pre_e, train = True)
    visualise(pre_e, train = False)
    if not args.visualise_only:
        eval(pre_e, train = True)
        eval(pre_e, train = False)
    # plt.figure()
    # plt.title('Test Loss')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.savefig('images/test_loss.png')
    if not args.visualise_only:
        with open('status.txt', 'w') as f:
            f.write('1')
    os._exit(0)

for e in range(EPOCHS):
    if pre_e > 0:
        pre_e -= 1
        continue
    loss, ssim = train(e)
    losses.append([loss, ssim])
    tloss, tssim = eval(e, train = False)
    test_losses.append([tloss, tssim])
    if args.neptune_log:
        run["train/loss"].log(loss)
        run["train/ssim"].log(ssim)
        run["test/loss"].log(tloss)
        run["test/ssim"].log(tssim)

    dic = {}
    dic['e'] = e+1
    dic['model'] = model.state_dict()
    dic['optimiser'] = optimiser.state_dict()
    dic['losses'] = losses
    dic['test_losses'] = test_losses
    
    torch.save(dic, checkpoint_path + 'checkpoint_{}.pth'.format(model_state))
    torch.save({'state': model_state}, checkpoint_path + 'state.pth')
    # model_state += 1
    print('Saving model after {} Epochs\n'.format(e+1), flush = True)
# eval(100, train = True)
# eval(100, train = False)