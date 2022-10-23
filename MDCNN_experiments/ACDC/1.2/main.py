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
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms, models, datasets
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('/root/Cardiac-MRI-Reconstrucion/')
os.environ['display'] = 'localhost:14.0'

from utils.MDCNN import MDCNN
from utils.myDatasets import ACDC
from utils.Trainers.DDP_MDCNNTrainer import Trainer

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def cleanup():
    dist.destroy_process_group()

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action = 'store_true')
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--test_only', action = 'store_true')
parser.add_argument('--visualise_only', action = 'store_true')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--port', type = int, default = 12355)
parser.add_argument('--state', type = int, default = -1)
parser.add_argument('--gpu', nargs='+', type = int, default = [-1])
# parser.add_argument('--gpu', type = int, default = '-1')
args = parser.parse_args()
seed_torch(args.seed)

parameters = {}
parameters['train_batch_size'] = 150
parameters['test_batch_size'] = 150
parameters['lr'] = 1e-3
parameters['num_epochs'] = 100
parameters['train_test_split'] = 0.8
parameters['dataset_path'] = '../../../datasets/ACDC'
parameters['normalisation'] = True
parameters['image_resolution'] = 64
parameters['window_size'] = 7
parameters['FT_radial_sampling'] = 14
parameters['predicted_frame'] = 'middle'
parameters['num_coils'] = 8
parameters['dataloader_num_workers'] = 0
parameters['optimizer'] = 'SGD'
parameters['scheduler'] = 'StepLR'
parameters['optimizer_params'] = (0.9, 1e-4)
parameters['scheduler_params'] = {
    'step_size': parameters['num_epochs']//3,
    'gamma': 0.1,
    'verbose': True
}
parameters['loss_recon'] = 'L1'
parameters['loss_FT'] = 'None'
parameters['loss_reconstructed_FT'] = 'None'
parameters['train_losses'] = []
parameters['test_losses'] = []
parameters['beta1'] = 1
parameters['beta2'] = 0.5
parameters['Automatic_Mixed_Precision'] = False
parameters['GPUs'] = args.gpu
parameters['loss_params'] = {
    'SSIM_window': 11,
    'alpha_phase': 1,
    'alpha_amp': 1,
    'grayscale': True,
    'deterministic': False,
    'watson_pretrained': True,
} 
assert(parameters['predicted_frame']) in ['last', 'middle']
assert(parameters['optimizer']) in ['Adam', 'SGD']
assert(parameters['loss_recon']) in ['L1', 'L2', 'SSIM', 'MS_SSIM']
assert(parameters['loss_FT']) in ['Cosine-L1', 'Cosine-L2', 'Cosine-SSIM', 'Cosine-MS_SSIM', 'None']
assert(parameters['loss_reconstructed_FT']) in ['Cosine-L1', 'Cosine-L2', 'Cosine-SSIM', 'Cosine-MS_SSIM', 'None', 'Cosine-Watson']

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '{}'.format(args.port)
    if args.gpu[0] == -1:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

if args.gpu[0] == -1:
    device = torch.device("cpu")
else:

    device = torch.device("cuda:{}".format(args.gpu[0]) if torch.cuda.is_available() else "cpu")
SAVE_INTERVAL = 1
# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

checkpoint_path = './checkpoints/'
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
if not os.path.isdir('./results/'):
    os.mkdir('./results/')
if not os.path.isdir('./results/train'):
    os.mkdir('./results/train')
if not os.path.isdir('./results/test'):
    os.mkdir('./results/test')

def train_paradigm(rank, world_size, shared_data):
    torch.cuda.set_device(args.gpu[rank])
    setup(rank, world_size)
    proc_device = torch.device('cuda:{}'.format(args.gpu[rank]))

    trainset = ACDC(
                        parameters['dataset_path'], 
                        train = True, 
                        train_split = parameters['train_test_split'], 
                        norm = parameters['normalisation'], 
                        resolution = parameters['image_resolution'], 
                        window_size = parameters['window_size'], 
                        ft_num_radial_views = parameters['FT_radial_sampling'], 
                        predict_mode = parameters['predicted_frame'], 
                        num_coils = parameters['num_coils'],
                        blank = True
                    )
    testset = ACDC(
                        parameters['dataset_path'], 
                        train = False, 
                        train_split = parameters['train_test_split'], 
                        norm = parameters['normalisation'], 
                        resolution = parameters['image_resolution'], 
                        window_size = parameters['window_size'], 
                        ft_num_radial_views = parameters['FT_radial_sampling'], 
                        predict_mode = parameters['predicted_frame'], 
                        num_coils = parameters['num_coils'],
                        blank = True
                    )

    trainset.set_shared_lists(shared_data)
    testset.set_shared_lists(shared_data)

    trainset.rest_init()
    testset.rest_init()

    model = MDCNN(8,7).to(proc_device)

    if args.resume:
        model_state = torch.load(checkpoint_path + 'state.pth', map_location = torch.device('cpu'))['state']
        if (not args.state == -1):
            model_state = args.state
        if rank == 0:
            print('Loading checkpoint at model state {}'.format(model_state), flush = True)
        dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state),map_location = proc_device)
        pre_e = dic['e']
        model.load_state_dict(dic['model'])
        opt_dict = dic['optim']
        scaler_dict = dic['scaler']
        if parameters['scheduler'] != 'None':
            scheduler_dict = dic['scheduler']
        losses = dic['losses']
        test_losses = dic['test_losses']
        if rank == 0:
            print('Resuming Training after {} epochs'.format(pre_e), flush = True)
    else:
        model_state = 0
        pre_e =0
        losses = []
        test_losses = []
        opt_dict = None
        scaler_dict = None
        scheduler_dict = None
        if rank == 0:
            print('Starting Training', flush = True)

    model = DDP(model, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = True)
    trainer = Trainer(model, trainset, testset, parameters, proc_device, rank, world_size)
    if args.resume:
        trainer.optim.load_state_dict(opt_dict)
        trainer.scaler.load_state_dict(scaler_dict)
        if parameters['scheduler'] != 'None':
            trainer.scaler.load_state_dict(scaler_dict)

    for e in range(parameters['num_epochs']):
        if pre_e > 0:
            pre_e -= 1
            continue
        collected_train_losses = [torch.zeros(3,).to(proc_device) for _ in range(world_size)]
        collected_test_losses = [torch.zeros(3,).to(proc_device) for _ in range(world_size)]
        
        lossrecon, lossft, lossreconft = trainer.train(e)
        dist.all_gather(collected_train_losses, torch.tensor([lossrecon, lossft, lossreconft]).to(proc_device))
        if rank == 0:
            avg_train_recon_loss = sum([x[0] for x in collected_train_losses]).item()
            avg_train_ft_loss = sum([x[1] for x in collected_train_losses]).item()
            avg_train_recon_ft_loss = sum([x[2] for x in collected_train_losses]).item()
            print('Average Train Recon Loss for Epoch {} = {}' .format(e, avg_train_recon_loss), flush = True)
            if trainer.criterion_FT is not None:
                print('Average Train FT Loss for Epoch {} = {}' .format(e, avg_train_ft_loss), flush = True)
            if trainer.criterion_reconFT is not None:
                print('Average Train Recon FT Loss for Epoch {} = {}' .format(e, avg_train_recon_ft_loss), flush = True)

        test_lossrecon, test_lossft, test_lossreconft = trainer.evaluate(e, train = False)
        dist.all_gather(collected_test_losses, torch.tensor([test_lossrecon, test_lossft, test_lossreconft]).to(proc_device))
        if rank == 0:
            avg_test_recon_loss = sum([x[0] for x in collected_test_losses]).item()
            avg_test_ft_loss = sum([x[1] for x in collected_test_losses]).item()
            avg_test_recon_ft_loss = sum([x[2] for x in collected_test_losses]).item()
            print('Test Loss After {} Epochs:'.format(e), flush = True)
            print('Recon Loss = {}'.format(avg_test_recon_loss), flush = True)
            if trainer.criterion_FT is not None:
                print('FT Loss = {}'.format(avg_test_ft_loss), flush = True)
            if trainer.criterion_reconFT is not None:
                print('Recon FT Loss = {}'.format(avg_test_recon_ft_loss), flush = True)

        if rank == 0:
            losses.append((avg_train_recon_loss, avg_train_ft_loss, avg_train_recon_ft_loss))
            test_losses.append((avg_test_recon_loss, avg_test_ft_loss, avg_test_recon_ft_loss))

            parameters['train_losses'] = losses
            parameters['test_losses'] = test_losses

            dic = {}
            dic['e'] = e+1
            dic['model'] = trainer.model.module.state_dict()
            dic['optim'] = trainer.optim.state_dict()
            if parameters['scheduler'] != 'None':
                dic['scheduler'] = trainer.scheduler.state_dict()
            dic['losses'] = losses
            dic['test_losses'] = test_losses
            dic['scaler'] = trainer.scaler.state_dict()
            if (e+1) % SAVE_INTERVAL == 0:
                torch.save(dic, checkpoint_path + 'checkpoint_{}.pth'.format(model_state))
                torch.save({'state': model_state}, checkpoint_path + 'state.pth')
                # model_state += 1
                print('Saving model after {} Epochs\n\n'.format(e+1), flush = True)

    cleanup()

def test_paradigm(rank, world_size, shared_data):
    torch.cuda.set_device(args.gpu[rank])
    setup(rank, world_size)
    proc_device = torch.device('cuda:{}'.format(args.gpu[rank]))

    trainset = ACDC(
                        parameters['dataset_path'], 
                        train = True, 
                        train_split = parameters['train_test_split'], 
                        norm = parameters['normalisation'], 
                        resolution = parameters['image_resolution'], 
                        window_size = parameters['window_size'], 
                        ft_num_radial_views = parameters['FT_radial_sampling'], 
                        predict_mode = parameters['predicted_frame'], 
                        num_coils = parameters['num_coils'],
                        blank = True
                    )
    testset = ACDC(
                        parameters['dataset_path'], 
                        train = False, 
                        train_split = parameters['train_test_split'], 
                        norm = parameters['normalisation'], 
                        resolution = parameters['image_resolution'], 
                        window_size = parameters['window_size'], 
                        ft_num_radial_views = parameters['FT_radial_sampling'], 
                        predict_mode = parameters['predicted_frame'], 
                        num_coils = parameters['num_coils'],
                        blank = True
                    )

    trainset.set_shared_lists(shared_data)
    testset.set_shared_lists(shared_data)

    trainset.rest_init()
    testset.rest_init()

    model = MDCNN(8,7).to(proc_device)

    if args.resume:
        model_state = torch.load(checkpoint_path + 'state.pth', map_location = torch.device('cpu'))['state']
        if (not args.state == -1):
            model_state = args.state
        if rank == 0:
            print('Loading checkpoint at model state {}'.format(model_state), flush = True)
        dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state), map_location = proc_device)
        pre_e = dic['e']
        model.load_state_dict(dic['model'])
        losses = dic['losses']
        test_losses = dic['test_losses']
        if rank == 0:
            print('Resuming Training after {} epochs'.format(pre_e), flush = True)
    else:
        model_state = 0
        pre_e =0
        losses = []
        test_losses = []
        opt_dict = None
        scaler_dict = None
        scheduler_dict = None
        if rank == 0:
            print('Starting Training', flush = True)

    model = DDP(model, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = True)
    trainer = Trainer(model, trainset, testset, parameters, proc_device, rank, world_size)
    
    if not args.visualise_only:
        test_lossrecon, test_lossft, test_lossreconft = trainer.evaluate(pre_e, train = False)
        collected_test_losses = [torch.zeros(3,).to(proc_device) for _ in range(world_size)]
        dist.all_gather(collected_test_losses, torch.tensor([test_lossrecon, test_lossft, test_lossreconft]).to(proc_device))
        if rank == 0:
            avg_test_recon_loss = sum([x[0] for x in collected_test_losses])
            avg_test_ft_loss = sum([x[1] for x in collected_test_losses])
            avg_test_recon_ft_loss = sum([x[2] for x in collected_test_losses])
            print('Test Loss After {} Epochs:'.format(pre_e), flush = True)
            print('Recon Loss = {}'.format(avg_test_recon_loss), flush = True)
            if trainer.criterion_FT is not None:
                print('FT Loss = {}'.format(avg_test_ft_loss), flush = True)
            if trainer.criterion_reconFT is not None:
                print('Recon FT Loss = {}'.format(avg_test_recon_ft_loss), flush = True)
        if not args.test_only:
            train_lossrecon, train_lossft, train_lossreconft = trainer.evaluate(pre_e, train = True)
            collected_train_losses = [torch.zeros(3,).to(proc_device) for _ in range(world_size)]
            dist.all_gather(collected_train_losses, torch.tensor([train_lossrecon, train_lossft, train_lossreconft]).to(proc_device))
            if rank == 0:
                avg_train_recon_loss = sum([x[0] for x in collected_train_losses])
                avg_train_ft_loss = sum([x[1] for x in collected_train_losses])
                avg_train_recon_ft_loss = sum([x[2] for x in collected_train_losses])
                print('Test Loss After {} Epochs:'.format(pre_e), flush = True)
                print('Recon Loss = {}'.format(avg_train_recon_loss), flush = True)
                if trainer.criterion_FT is not None:
                    print('FT Loss = {}'.format(avg_train_ft_loss), flush = True)
                if trainer.criterion_reconFT is not None:
                    print('Recon FT Loss = {}'.format(avg_train_recon_ft_loss), flush = True)
    if rank == 0:
        trainer.visualise(pre_e, train = False)
        trainer.visualise(pre_e, train = True)
        plt.figure()
        plt.title('Train Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(losses)), [x[0] for x in losses], label = 'Recon Loss: {}'.format(parameters['loss_recon']), color = 'b')
        if parameters['loss_FT'] != 'None':
            plt.plot(range(len(losses)), [x[1] for x in losses], label = 'FT Loss: {}'.format(parameters['loss_FT']), color = 'r')
        if parameters['loss_reconstructed_FT'] != 'None':
            plt.plot(range(len(losses)), [x[2] for x in losses], label = 'Recon FT Loss: {}'.format(parameters['loss_reconstructed_FT']), color = 'g')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('results/train_loss.png')
        plt.figure()
        plt.title('Test Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(test_losses)), [x[0] for x in test_losses], label = 'Recon Loss: {}'.format(parameters['loss_recon']), color = 'b')
        if parameters['loss_FT'] != 'None':
            plt.plot(range(len(test_losses)), [x[1] for x in test_losses], label = 'FT Loss: {}'.format(parameters['loss_FT']), color = 'r')
        if parameters['loss_reconstructed_FT'] != 'None':
            plt.plot(range(len(test_losses)), [x[2] for x in test_losses], label = 'Recon FT Loss: {}'.format(parameters['loss_reconstructed_FT']), color = 'g')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('results/test_loss.png')
        plt.close('all')
        with open('status.txt', 'w') as f:
            f.write('1')
    cleanup()

if __name__ == '__main__':
    world_size = len(args.gpu) 
    trainset = ACDC(
                        parameters['dataset_path'], 
                        train = True, 
                        train_split = parameters['train_test_split'], 
                        norm = parameters['normalisation'], 
                        resolution = parameters['image_resolution'], 
                        window_size = parameters['window_size'], 
                        ft_num_radial_views = parameters['FT_radial_sampling'], 
                        predict_mode = parameters['predicted_frame'], 
                        num_coils = parameters['num_coils']
                    )
    shared_data = trainset.get_shared_lists()
    if args.eval:
        mp.spawn(
            test_paradigm,
            args=[world_size, shared_data],
            nprocs=world_size
        )
        os._exit(0)
    mp.spawn(
        train_paradigm,
        args=[world_size, shared_data],
        nprocs=world_size
    )