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

sys.path.append('../../../')
os.environ['display'] = 'localhost:14.0'

from utils.MDCNN import MDCNN
from utils.myDatasets import ToyVideoSet

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


if args.gpu == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 50
TEST_BATCH_SIZE = 50
lr = 1e-3
NUM_EPOCHS = 1000
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

trainset = ToyVideoSet('../../../datasets/toy_data',train = True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle = True)
traintestloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle = False)
testset = ToyVideoSet('../../../datasets/toy_data',train = False)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle = False)

model = MDCNN(1,7).to(device)
optim = optim.Adam(model.parameters(), lr=lr, betas=(.5, 0.999))
criterion = nn.MSELoss().to(device)

if args.resume:
    model_state = torch.load(checkpoint_path + 'state.pth', map_location = device)['state']
    if (not args.state == -1):
        model_state = args.state
    print('Loading checkpoint at model state {}'.format(model_state), flush=True)
    dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state), map_location = device)
    pre_e = dic['e']
    model.load_state_dict(dic['model'])
    optim.load_state_dict(dic['optim'])
    losses = dic['losses']
    test_losses = dic['test_losses']
    print('Resuming Training after {} epochs'.format(pre_e))
else:
    model_state = 0
    pre_e =0
    losses = []
    test_losses = []
    print('Starting Training')

def train(epoch):
    totloss = 0
    for i, (indices, fts, fts_masked, targets) in tqdm(enumerate(trainloader), total = len(trainloader), desc = "[{}] | Epoch {}".format(os.getpid(), epoch)):
        optim.zero_grad()
        preds = model(fts_masked.to(device)) # B, 1, X, Y
        loss = criterion(preds.to(device), targets.to(device))
        loss.backward()
        optim.step()
        totloss += loss.item()

    print('Average Loss for Epoch {} = {}' .format(epoch, totloss/(len(trainloader))))
    return totloss/(len(trainloader))

def evaluate(epoch, train = False):
    if train:
        dloader = traintestloader
        dset = trainset
        dstr = 'Train'
    else:
        dloader = testloader
        dset = testset
        dstr = 'Test'
    totlossL2 = 0
    with torch.no_grad():
        for i, (indices, fts, fts_masked, targets) in tqdm(enumerate(dloader), total = len(dloader), desc = "Testing after Epoch {} on {}set".format(epoch, dstr)):
            preds = model(fts_masked.to(device))
            totlossL2 += criterion(preds.to(device), targets.to(device)).item()

    print('{} Loss After {} Epochs:'.format(dstr, epoch))
    print('L2 Loss = {}'.format(totlossL2/(len(dloader))))
    return totlossL2/(len(dloader))

def visualise(epoch, num_plots = 10, train = False):
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
    print('Saving plots for {} data'.format(dstr))
    totloss = 0
    with torch.no_grad():
        for i, (indices, fts, fts_masked, targets) in enumerate(dloader):
            preds = model(fts_masked.to(device))
            break
        for i in range(num_plots):
            targi = targets[i].squeeze().cpu().numpy()
            predi = preds[i].squeeze().cpu().numpy()
            fig = plt.figure(figsize = (8,8))
            plt.subplot(2,2,1)
            ft = torch.complex(fts_masked[i,0,3,:,:,0],fts_masked[i,0,3,:,:,1])
            plt.imshow(ft.abs(), cmap = 'gray')
            plt.title('Undersampled FFT Frame')
            plt.subplot(2,2,2)
            plt.imshow(torch.fft.ifft2(torch.fft.ifftshift(ft.exp())).real, cmap = 'gray')
            plt.title('IFFT of the Input')
            plt.subplot(2,2,3)
            plt.imshow(predi, cmap = 'gray')
            plt.title('Our Predicted Frame')
            plt.subplot(2,2,4)
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
    plt.plot(losses, label = 'Train Loss', color = 'b')
    plt.plot(test_losses, label = 'Test Loss L2', color = 'r')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('results/loss.png')
    plt.close('all')
    with open('status.txt', 'w') as f:
        f.write('1')
    os._exit(0)


for e in range(NUM_EPOCHS):
    if pre_e > 0:
        pre_e -= 1
        continue
    loss = train(e)
    losses.append(loss)
    loss = evaluate(e, train = False)
    test_losses.append(loss)

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
        print('Saving model after {} Epochs'.format(e+1))
