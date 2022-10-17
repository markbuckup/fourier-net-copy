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
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms, models, datasets

sys.path.append('/root/Cardiac-MRI-Reconstrucion/')
os.environ['display'] = 'localhost:14.0'

from utils.MDCNN import MDCNN
from utils.myDatasets import ACDC
from utils.functions import fetch_loss_function

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action = 'store_true')
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--state', type = int, default = -1)
parser.add_argument('--gpu', type = int, default = '-1')
args = parser.parse_args()
seed_torch(args.seed)

parameters = {}
parameters['train_batch_size'] = 70
parameters['test_batch_size'] = 70
parameters['lr'] = 1e-3
parameters['num_epochs'] = 1000
parameters['train_test_split'] = 0.8
parameters['normalisation'] = True
parameters['image_resolution'] = 128
parameters['window_size'] = 7
parameters['FT_radial_sampling'] = 14
parameters['predicted_frame'] = 'middle'
parameters['num_coils'] = 8
parameters['optimizer'] = 'Adam'
parameters['optimizer_params'] = (0.5, 0.999)
parameters['loss'] = 'Cosine-Watson'
parameters['loss_FT'] = 'None'
parameters['loss_reconstructed_FT'] = 'Cosine-L1'
parameters['train_losses'] = []
parameters['test_losses'] = []
parameters['beta1'] = 1
parameters['beta2'] = 1
if args.eval:
    parameters['GPUS'] = [1]
else:
    parameters['GPUS'] = [2,3]
parameters['loss_params'] = {
    # 'SSIM_window': 11,
    'alpha_phase': 1,
    'alpha_amp': 1,
    'grayscale': True,
    'deterministic': False,
    'watson_pretrained': True,
} 
assert(parameters['predicted_frame']) in ['last', 'middle']
assert(parameters['optimizer']) in ['Adam', 'SGD']
assert(parameters['loss']) in ['L1', 'L2', 'SSIM', 'MS_SSIM', 'Cosine-Watson']
assert(parameters['loss_FT']) in ['Cosine-L1', 'Cosine-L2', 'Cosine-SSIM', 'Cosine-MS_SSIM', 'None']
assert(parameters['loss_reconstructed_FT']) in ['Cosine-L1', 'Cosine-L2', 'Cosine-SSIM', 'Cosine-MS_SSIM', 'None']

if args.gpu == -1:
    device = torch.device("cpu")
else:

    device = torch.device("cuda:{}".format(parameters['GPUS'][0]) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda".format(args.gpu) if torch.cuda.is_available() else "cpu")
SAVE_INTERVAL = 1
torch.autograd.set_detect_anomaly(True)

checkpoint_path = './checkpoints/'
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
if not os.path.isdir('./results/'):
    os.mkdir('./results/')
if not os.path.isdir('./results/train'):
    os.mkdir('./results/train')
if not os.path.isdir('./results/test'):
    os.mkdir('./results/test')

trainset = ACDC(
                    '../../../datasets/ACDC', 
                    train = True, 
                    train_split = parameters['train_test_split'], 
                    norm = parameters['normalisation'], 
                    resolution = parameters['image_resolution'], 
                    window_size = parameters['window_size'], 
                    ft_num_radial_views = parameters['FT_radial_sampling'], 
                    predict_mode = parameters['predicted_frame'], 
                    num_coils = parameters['num_coils']
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
testset = ACDC(
                    '../../../datasets/ACDC', 
                    train = False, 
                    train_split = parameters['train_test_split'], 
                    norm = parameters['normalisation'], 
                    resolution = parameters['image_resolution'], 
                    window_size = parameters['window_size'], 
                    ft_num_radial_views = parameters['FT_radial_sampling'], 
                    predict_mode = parameters['predicted_frame'], 
                    num_coils = parameters['num_coils']
                )
testloader = torch.utils.data.DataLoader(
                    testset, 
                    batch_size=parameters['test_batch_size'],
                    shuffle = False)

model = nn.DataParallel(MDCNN(8,7), device_ids = parameters['GPUS'])
model.to(f'cuda:{model.device_ids[0]}')

if parameters['optimizer'] == 'Adam':
    optim = optim.Adam(model.parameters(), lr=parameters['lr'], betas=parameters['optimizer_params'])
elif parameters['optimizer'] == 'SGD':
    mom, wt_dec = parameters['optimizer_params']
    optim = optim.SGD(model.parameters(), lr=parameters['lr'], momentum = mom, weight_decay = wt_dec)

criterion = fetch_loss_function(parameters['loss'], device, parameters['loss_params']).to(device)
criterion_FT = fetch_loss_function(parameters['loss_FT'], device, parameters['loss_params'])
# if criterion_FT is not None:
#     criterion_FT = criterion_FT.to(device)
criterion_reconFT = fetch_loss_function(parameters['loss_reconstructed_FT'], device, parameters['loss_params'])
# if criterion_reconFT is not None:
#     criterion_reconFT = criterion_reconFT.to(device)

if args.resume:
    model_state = torch.load(checkpoint_path + 'state.pth', map_location = device)['state']
    if (not args.state == -1):
        model_state = args.state
    print('Loading checkpoint at model state {}'.format(model_state), flush = True)
    dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state), map_location = device)
    pre_e = dic['e']
    model.load_state_dict(dic['model'])
    optim.load_state_dict(dic['optim'])
    losses = dic['losses']
    test_losses = dic['test_losses']
    print('Resuming Training after {} epochs'.format(pre_e), flush = True)
else:
    model_state = 0
    pre_e =0
    losses = []
    test_losses = []
    print('Starting Training', flush = True)

def train(epoch):
    totlossrecon = 0
    totlossft = 0
    totlossreconft = 0
    beta1 = parameters['beta1']
    beta2 = parameters['beta2']
    for i, (indices, fts, fts_masked, targets, target_fts) in tqdm(enumerate(trainloader), total = len(trainloader), desc = "[{}] | Epoch {}".format(os.getpid(), epoch)):
        optim.zero_grad()
        # print('Computing forward', fts_masked.shape, flush = True)
        # t = time.process_time()
        ft_preds, preds = model(fts_masked.to(device)) # B, 1, X, Y
        # print('Computed forward', time.process_time()-t, flush = True)
        # print('Computing loss', flush = True)
        # t = time.process_time()
        loss_recon = criterion(preds.to(device), targets.to(device))
        # print('Computed loss', time.process_time()-t, flush = True)
        loss_ft = torch.tensor([0]).to(device)
        loss_reconft = torch.tensor([0]).to(device)
        if criterion_FT is not None:
            loss_ft = criterion_FT(ft_preds.to(device), target_fts.to(device))
        if criterion_reconFT is not None:
            predfft = torch.fft.fft2(preds.to(device)).log()
            predfft = torch.stack((predfft.real, predfft.imag),-1)
            targetfft = torch.fft.fft2(targets.to(device)).log()
            targetfft = torch.stack((targetfft.real, targetfft.imag),-1)
            loss_reconft = criterion_reconFT(predfft, targetfft)
        loss = loss_recon + beta1*loss_ft + beta2*loss_reconft
        # print('Computing backward', fts_masked.shape, flush = True)
        # t = time.process_time()      
        loss.backward()
        optim.step()
        # print('Computed backward', time.process_time()-t, flush = True)
        totlossrecon += loss_recon.item()
        totlossft += loss_ft.item()
        totlossreconft += loss_reconft.item()

    print('Average Recon Loss for Epoch {} = {}' .format(epoch, totlossrecon/(len(trainloader))), flush = True)
    if criterion_FT is not None:
        print('Average FT Loss for Epoch {} = {}' .format(epoch, totlossft/(len(trainloader))), flush = True)
    if criterion_reconFT is not None:
        print('Average Recon FT Loss for Epoch {} = {}' .format(epoch, totlossreconft/(len(trainloader))), flush = True)
    return totlossrecon/(len(trainloader)), totlossft/(len(trainloader)), totlossreconft/(len(trainloader))

def evaluate(epoch, train = False):
    if train:
        dloader = traintestloader
        dset = trainset
        dstr = 'Train'
    else:
        dloader = testloader
        dset = testset
        dstr = 'Test'
    totlossrecon = 0
    totlossft = 0
    totlossreconft = 0
    with torch.no_grad():
        for i, (indices, fts, fts_masked, targets, target_fts) in tqdm(enumerate(dloader), total = len(dloader), desc = "Testing after Epoch {} on {}set".format(epoch, dstr)):
            ft_preds, preds = model(fts_masked.to(device))
            totlossrecon += criterion(preds.to(device), targets.to(device)).item()
            if criterion_FT is not None:
                totlossft += criterion_FT(ft_preds.to(device), target_fts.to(device)).item()
            if criterion_reconFT is not None:
                predfft = torch.fft.fft2(preds.to(device)).log()
                predfft = torch.stack((predfft.real, predfft.imag),-1)
                targetfft = torch.fft.fft2(targets.to(device)).log()
                targetfft = torch.stack((targetfft.real, targetfft.imag),-1)
                totlossreconft += criterion_reconFT(predfft, targetfft).item()

    print('{} Loss After {} Epochs:'.format(dstr, epoch), flush = True)
    print('Loss = {}'.format(totlossrecon/(len(dloader))), flush = True)
    if criterion_FT is not None:
        print('Loss = {}'.format(totlossft/(len(dloader))), flush = True)
    if criterion_reconFT is not None:
        print('Loss = {}'.format(totlossreconft/(len(dloader))), flush = True)
    return totlossrecon/(len(dloader)), totlossft/(len(dloader)), totlossreconft/(len(dloader))

def visualise(epoch, num_plots = parameters['test_batch_size'], train = False):
    if train:
        dloader = traintestloader
        dset = trainset
        dstr = 'Train'
        path = './results/train'
    else:
        dloader = testloader
        dset = testset
        dstr = 'Test'
        path = './results/test'
    print('Saving plots for {} data'.format(dstr), flush = True)
    totloss = 0
    with torch.no_grad():
        for i, (indices, fts, fts_masked, targets, target_fts) in enumerate(dloader):
            ft_preds, preds = model(fts_masked.to(device))
            break
        for i in range(num_plots):
            targi = targets[i].squeeze().cpu().numpy()
            predi = preds[i].squeeze().cpu().numpy()
            # fig = plt.figure(figsize = (8,8))
            # plt.subplot(2,2,1)
            # ft = torch.complex(fts_masked[i,0,3,:,:,0],fts_masked[i,0,3,:,:,1])
            # plt.imshow(ft.abs(), cmap = 'gray')
            # plt.title('Undersampled FFT Frame')
            # plt.subplot(2,2,2)
            # plt.imshow(torch.fft.ifft2(torch.fft.ifftshift(ft.exp())).real, cmap = 'gray')
            # plt.title('IFFT of the Input')
            # plt.subplot(2,2,3)
            # plt.imshow(predi, cmap = 'gray')
            # plt.title('Our Predicted Frame')
            # plt.subplot(2,2,4)
            # plt.imshow(targi, cmap = 'gray')
            # plt.title('Actual Frame')
            # plt.suptitle("{} data window index {}".format(dstr, indices[i]))
            # plt.savefig(os.path.join(path, '{}_result_epoch{}_{}'.format(dstr, epoch, i)))
            # plt.close('all')
            fig = plt.figure(figsize = (8,4))
            # plt.subplot(2,2,1)
            # ft = torch.complex(fts_masked[i,0,3,:,:,0],fts_masked[i,0,3,:,:,1])
            # plt.imshow(ft.abs(), cmap = 'gray')
            # plt.title('Undersampled FFT Frame')
            # plt.subplot(2,2,2)
            # plt.imshow(torch.fft.ifft2(torch.fft.ifftshift(ft.exp())).real, cmap = 'gray')
            # plt.title('IFFT of the Input')
            plt.subplot(1,2,1)
            plt.imshow(predi, cmap = 'gray')
            plt.title('Our Predicted Frame')
            plt.subplot(1,2,2)
            plt.imshow(targi, cmap = 'gray')
            plt.title('Actual Frame')
            plt.suptitle("{} data window index {}".format(dstr, indices[i]))
            plt.savefig(os.path.join(path, '{}_result_epoch{}_{}'.format(dstr, epoch, i)))
            plt.close('all')

if args.eval:
    evaluate(pre_e, train = False)
    evaluate(pre_e, train = True)
    visualise(pre_e, train = False)
    visualise(pre_e, train = True)
    plt.figure()
    plt.title('Train Loss')
    plt.plot([x[0] for x in losses], label = 'Recon Loss', color = 'b')
    if criterion_FT is not None:
        plt.plot([x[1] for x in losses], label = 'FT Loss', color = 'r')
    if criterion_reconFT is not None:
        plt.plot([x[2] for x in losses], label = 'Recon FT Loss', color = 'g')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('results/train_loss.png')
    plt.figure()
    plt.title('Test Loss')
    plt.plot([x[0] for x in test_losses], label = 'Recon Loss', color = 'b')
    if criterion_FT is not None:
        plt.plot([x[1] for x in test_losses], label = 'FT Loss', color = 'r')
    if criterion_reconFT is not None:
        plt.plot([x[2] for x in test_losses], label = 'Recon FT Loss', color = 'g')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('results/test_loss.png')
    plt.close('all')
    with open('status.txt', 'w') as f:
        f.write('1')
    os._exit(0)


for e in range(parameters['num_epochs']):
    if pre_e > 0:
        pre_e -= 1
        continue
    lossrecon, lossft, lossreconft = train(e)
    losses.append((lossrecon, lossft, lossreconft))
    lossrecon, lossft, lossreconft = evaluate(e, train = False)
    test_losses.append((lossrecon, lossft, lossreconft))

    parameters['train_losses'] = losses
    parameters['test_losses'] = test_losses

    dic = {}
    dic['e'] = e+1
    dic['model'] = model.state_dict()
    dic['optim'] = optim.state_dict()
    dic['losses'] = losses
    dic['test_losses'] = test_losses


    if (e+1) % SAVE_INTERVAL == 0:
        torch.save(dic, checkpoint_path + 'checkpoint_{}.pth'.format(model_state))
        torch.save({'state': model_state}, checkpoint_path + 'state.pth')
        # model_state += 1
        print('Saving model after {} Epochs'.format(e+1), flush = True)
