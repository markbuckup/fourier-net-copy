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
from utils.models.convLSTM import convLSTM_theta as Model
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

EPS = 1e-8
CEPS = torch.complex(torch.tensor(EPS),torch.tensor(EPS))

EPOCHS = 500
parameters = {}
parameters['lr'] = 3e-4
parameters['descrption'] = 'Base experiment - parametrised phase as theta'
parameters['train_batch_size'] = 23
parameters['test_batch_size'] = 23
parameters['num_views'] = 14
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

model = Model(tanh_mode = False, sigmoid_mode = True).to(device)
optimiser = optim.Adam(list(model.parameters()), lr = parameters['lr'])
criterionL1 = nn.L1Loss().to(device)
criterionCos = nn.CosineSimilarity(dim = 5)
SSIM = kornia.metrics.SSIM(11)

def myimshow(x, cmap = 'gray'):
    percentile_95 = np.percentile(x, 95)
    percentile_5 = np.percentile(x, 5)
    x[x > percentile_95] = percentile_95
    x[x < percentile_5] = percentile_5
    x = x - x.min()
    x = x/ (x.max() + EPS)
    plt.imshow(x, cmap = cmap)

def train(epoch):
    loss_av_mag = 0
    loss_av_phase = 0
    ssim_score = 0
    coil_mask_coll = get_golden_bars(resolution = 64)
    for i, (indices, data) in tqdm(enumerate(trainloader), total = len(trainloader), desc = "[{}] | Training Ep {}".format(os.getpid(), epoch), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}"):
        optimiser.zero_grad(set_to_none = True)
        batch, num_frames, chan, numr, numc = data.shape
        coil_mask = coil_mask_coll[(torch.arange(parameters['num_views']*num_frames)+i)%376,:,:]
        coil_mask = coil_mask.reshape(num_frames,parameters['num_views'],numr,numc).sum(1).sign().unsqueeze(0).unsqueeze(2).to(device)
        inpt = (torch.fft.fftshift(torch.fft.fft2(data.to(device)), dim = (-2,-1))+CEPS)
        inpt_mag_log = inpt.log().real
        inpt_phase = inpt / inpt_mag_log.exp()
        inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)

        model.train()
        ans_phase, ans_mag_log = model(inpt,coil_mask, device)

        loss1 = criterionL1(ans_mag_log, inpt_mag_log)
        loss2 = (1 - criterionCos(ans_phase, inpt_phase)).mean()
        loss = loss1 + loss2
        with torch.no_grad():
            pred_ft = ans_phase*ans_mag_log.unsqueeze(-1).exp()
            pred_ft = torch.complex(pred_ft[:,:,:,:,:,0],pred_ft[:,:,:,:,:,1])
            predr = torch.fft.ifft2(torch.fft.ifftshift(pred_ft.exp(), dim = (-2,-1))).real
        
        # loss = criterionL1(pred, data.to(device))

        loss.backward()
        optimiser.step()

        loss_av_mag += loss1.item()/len(trainloader)
        loss_av_phase += loss2.item()/len(trainloader)
        ss1 = SSIM(predr.reshape(batch*num_frames,chan,numr,numc), data.reshape(batch*num_frames,chan,numr,numc).to(device))
        ss1 = ss1.reshape(ss1.shape[0], -1).mean(1)
        ssim_score += (ss1/(len(trainset)*num_frames)).sum().item()

    print('Train Mag loss for epoch {} = {}'.format(epoch, loss_av_mag), flush = True)
    print('Train Phase loss for epoch {} = {}'.format(epoch, loss_av_phase), flush = True)
    print('Train SSIM for epoch {} = {}'.format(epoch, ssim_score), flush = True)
    return loss_av_mag, loss_av_phase, ssim_score

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
        loss_av_mag = 0
        loss_av_phase = 0
        ssim_score = 0
        coil_mask_coll = get_golden_bars(resolution = 64)
        for i, (indices, data) in tqdm(enumerate(dloader), total = len(dloader), desc = "[{}] | Testing Ep {}".format(os.getpid(), epoch), bar_format="{desc} | {percentage:3.0f}%|{bar:10}{r_bar}"):
            batch, num_frames, chan, numr, numc = data.shape
            coil_mask = coil_mask_coll[(torch.arange(parameters['num_views']*num_frames)+i)%376,:,:]
            coil_mask = coil_mask.reshape(num_frames,parameters['num_views'],numr,numc).sum(1).sign().unsqueeze(0).unsqueeze(2).to(device)
            inpt = (torch.fft.fftshift(torch.fft.fft2(data.to(device)), dim = (-2,-1))+CEPS)
            inpt_mag_log = inpt.log().real
            inpt_phase = inpt / inpt_mag_log.exp()
            inpt_phase = torch.stack((inpt_phase.real, inpt_phase.imag),-1)

            model.eval()
            ans_phase, ans_mag_log = model(inpt,coil_mask, device)
            pred_ft = ans_phase*ans_mag_log.unsqueeze(-1).exp()
            pred_ft = torch.complex(pred_ft[:,:,:,:,:,0],pred_ft[:,:,:,:,:,1])

            loss1 = criterionL1(ans_mag_log, inpt_mag_log)
            loss2 = (1 - criterionCos(ans_phase, inpt_phase)).mean()
            loss = loss1 + loss2
            predr = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).real
            
            # loss = criterionL1(pred, data.to(device))

            loss_av_mag += loss1.item()/len(dloader)
            loss_av_phase += loss2.item()/len(dloader)
            ss1 = SSIM(predr.reshape(batch*num_frames,chan,numr,numc), data.reshape(batch*num_frames,chan,numr,numc).to(device))
            ss1 = ss1.reshape(ss1.shape[0], -1).mean(1)
            ssim_score += (ss1/(len(dset)*num_frames)).sum().item()

    print('Test Mag loss for epoch {} = {}'.format(epoch, loss_av_mag), flush = True)
    print('Test Phase loss for epoch {} = {}'.format(epoch, loss_av_phase), flush = True)
    print('Test SSIM for epoch {} = {}'.format(epoch, ssim_score), flush = True)
    return loss_av_mag, loss_av_phase, ssim_score

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
            
            model.eval()

            ans_phase, ans_mag_log = model(inpt,coil_mask, device)
            pred_ft = ans_phase*ans_mag_log.unsqueeze(-1).exp()
            pred_ft = torch.complex(pred_ft[:,:,:,:,:,0],pred_ft[:,:,:,:,:,1])

            predr = torch.fft.ifft2(torch.fft.ifftshift(pred_ft, dim = (-2,-1))).abs()
            # predr[:,:,:,:5,:5] = 0
            # predr[:,:,:,-5:,-5:] = 0
            # predr[:,:,:,-5:,:5] = 0
            # predr[:,:,:,:5,-5:] = 0
            target = torch.fft.ifft2(torch.fft.ifftshift(inpt, dim = (-2,-1))).real
            under_targ = torch.fft.ifft2(torch.fft.ifftshift(inpt*coil_mask, dim = (-2,-1))).real
            
            for i in range(pred_ft.shape[1]):
                fig = plt.figure(figsize = (12,8))
                plt.subplot(2,3,1)
                myimshow(((inpt+CEPS).log())[0,i,0,:,:].abs().cpu().numpy(), cmap = 'gray')
                plt.title('Complete FFT')
                plt.subplot(2,3,2)
                myimshow(((inpt+CEPS).log()*coil_mask)[0,i,0,:,:].abs().cpu().numpy(), cmap = 'gray')
                plt.title('Undersampled FFT')
                plt.subplot(2,3,3)
                myimshow((pred_ft+CEPS).log()[0,i,0,:,:].abs().cpu().numpy(), cmap = 'gray')
                plt.title('Predicted FFT')
                plt.subplot(2,3,4)
                myimshow(target.cpu()[0,i,0,:,:], cmap = 'gray')
                plt.title('Image from Complete FFT')
                plt.subplot(2,3,5)
                myimshow(under_targ.cpu()[0,i,0,:,:], cmap = 'gray')
                plt.title('Image from Undersampled FFT')
                plt.subplot(2,3,6)
                myimshow(predr[0,i,0,:,:].cpu().numpy(), cmap = 'gray')
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
    plt.title('Train Loss')
    plt.plot(range(len(losses)), [x[0] for x in losses], color = 'b', label = 'Train Mag Loss')
    plt.plot(range(len(losses)), [x[1] for x in losses], color = 'r', label = 'Train Phase Loss')
    plt.plot(range(len(losses)), [x[2] for x in losses], color = 'g', label = 'Train SSIM')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('images/train_loss.png')
    plt.figure()
    plt.title('Test Loss')
    plt.plot(range(len(test_losses)), [x[0] for x in test_losses], color = 'b', label = 'Test Mag Loss')
    plt.plot(range(len(test_losses)), [x[1] for x in test_losses], color = 'r', label = 'Test Phase Loss')
    plt.plot(range(len(test_losses)), [x[2] for x in test_losses], color = 'g', label = 'Test SSIM')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('images/test_loss.png')
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
    os._exit(0)

for e in range(EPOCHS):
    if pre_e > 0:
        pre_e -= 1
        continue
    loss_mag, loss_phase, ssim = train(e)
    losses.append([loss_mag, loss_phase, ssim])
    tloss_mag, tloss_phase, tssim = eval(e, train = False)
    test_losses.append([tloss_mag, tloss_phase, tssim])
    if args.neptune_log:
        run["train/loss_mag"].log(loss_mag)
        run["train/loss_phase"].log(loss_phase)
        run["train/ssim"].log(ssim)
        run["test/loss_mag"].log(tloss_mag)
        run["test/loss_phase"].log(tloss_phase)
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