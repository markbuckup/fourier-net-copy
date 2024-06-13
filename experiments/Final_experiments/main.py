import os
import gc
import sys
import PIL
import time
import cv2
import torch
import random
import pickle
import argparse
import numpy as np
import torchvision
import neptune as neptune
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from torch import nn, optim
from neptune.types import File
from torch.nn import functional as F
from torchvision import transforms, models, datasets
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('../../')
os.environ['display'] = 'localhost:14.0'
from utils.DDP_paradigms_LSTM_nufft import train_paradigm, test_paradigm

parser = argparse.ArgumentParser()
parser.add_argument('--resume', 
                            action = 'store_true', 
                            help = 'Resumes the code from the latest checkpoint'
                    )
parser.add_argument('--time_analysis',
                            action = 'store_true', 
                            help = 'Computes and prints the time taken per frame'
                    )
parser.add_argument('--resume_kspace', 
                            action = 'store_true', 
                            help = 'Resumes the Neural Network after the first 100 epochs, that is, before the image lstm starts training'
                    )
parser.add_argument('--eval', 
                            action = 'store_true', 
                            help = 'Evaluates the architecture - prints the SSIM and saves the predicted images'
                    )
parser.add_argument('--eval_on_real', 
                            action = 'store_true', 
                            help = '[Under development] Runs the evaluate script on real data'
                    )
parser.add_argument('--write_csv', 
                            action = 'store_true', 
                            help = '[Not Updated since October 2023] Stores the l1/l2/ssim scores for each frame into a csv'
                    )
parser.add_argument('--test_only', 
                            action = 'store_true', 
                            help = 'Evaluates the model only on the test data. Should be run along with the args.eval argument'
                    )
parser.add_argument('--visualise_only', 
                            action = 'store_true',
                            help = 'Stores the predicted images ONLY - no SSIM/L1/L2. Should be run along with the args.eval argument'
                    )
parser.add_argument('--numbers_only', 
                            action = 'store_true',
                            help = 'Prints the L1/L2/SSIM ONLY - no predicted images. Should be run along with the args.eval argument'
                    )
parser.add_argument('--unet_visual_only', 
                            action = 'store_true',
                            help = 'Stores only the final coil-combined predictions of the FOURIER-Net. Does not store the intermediate predctions by the KSpace RNN and the ILSTM'
                    )
parser.add_argument('--raw_visual_only', 
                            action = 'store_true',
                            help = 'Stores all the predictions as raw images - without and matplotlib gridding / figures. This is needed before generating the video'
                    )
parser.add_argument('--seed', 
                            type = int, 
                            default = 0,
                            help = 'Fix a seed so that the training is deterministic - retraining gives the same results'
                    )
parser.add_argument('--port', 
                            type = int, 
                            default = 12355,
                            help = 'Port for multi-gpu training - no need to touch this :)'
                    )
parser.add_argument('--run_id', 
                            type = str, 
                            required = True,
                            help = 'The only required parameter. Must have the path to an experiment folder that contains a parms.py file. A corresponding folder will be created on the NAS to store the checkpoints and the images.'
                    )
parser.add_argument('--state', 
                            type = int, 
                            default = -1,
                            help = 'Resumes the code from a particular "model state". Model states increase with epochs - you can resume from a previous epoch (instead of the latest epoch) by using this argument. This is useful in cases like - if the training loss becomes NAN while training, then we can resume from a previous state and try to diagnose this. Defaults to the latest checkpoint.'
                    )
parser.add_argument('--gpu', 
                            nargs='+', 
                            type = int, 
                            default = [-1],
                            help = 'The GPU ids to be used. Can be multiple integers. Defaults to cpu training'
                    )
parser.add_argument('--neptune_log', 
                            action = 'store_true',
                            help = 'If enabled, logs the training progress to neptune.ai'
                    )
parser.add_argument('--actual_data_path', 
                            default = '../../datasets/actual_data/data1.pth',
                            help = 'Path to the actual_data ".pth" dictionary'
                    )
parser.add_argument('--dataset_path', 
                            default = '/Data/ExtraDrive1/niraj/datasets/ACDC/',
                            help = 'Path to the dataset'
                    )
args = parser.parse_args()


# A few assertions for the arguments
not (args.eval_on_real and args.eval)
assert(not (args.raw_visual_only and args.unet_visual_only))
if args.write_csv:
    assert(args.eval)
if args.test_only:
    assert(args.eval)
if args.unet_visual_only:
    assert(args.eval and args.visualise_only)
if args.state != -1:
    assert(args.resume)
assert(not (args.visualise_only and args.numbers_only))
# Currently not supported on cpu
assert(args.gpu != [-1])
if args.eval:
    len(args.gpu == 1)


if args.run_id[-1] == '/':
    args.run_id = args.run_id[:-1]
sys.path.append('./{}/'.format(args.run_id))


# Import the parameters from the params.py
from params import parameters
if parameters['dataset'] == 'acdc':
    from utils.myDatasets.ACDC_radial import ACDC_radial as dataset
    

# If evaluating, then just compute the L1/L2/SSIM numbers on the last 30 frames
if args.eval:
    parameters['init_skip_frames'] = 90


# If evaluating, then fix the batch size to 1, since for evaluation, we use the entire video of 30 frames. 
# We do this so that we do not run out of GPU memory
if args.eval:
    parameters['train_batch_size'] = 1
    parameters['test_batch_size'] = 1


# Fix the seed during training
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(args.seed)
 

# A few hacks to speed up training
# This makes the code faster, but can make it indeterministic even if the seed has been fixed
torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)



# The "main" block
if __name__ == '__main__':
    world_size = len(args.gpu) 

    # Spawn the multiple children processes - one process per gpu
    # Each child is spawned with a function "train_paradigm" or "test_paradigm". This function is executed by each child when it spawns.
    # If evaluating - then spawn the children with test_paradigm
    # If training - then spawn the children with train_paradigm
    if args.eval or args.eval_on_real:
        mp.spawn(
            test_paradigm,
            args=[world_size, args, parameters],
            nprocs=world_size
        )
        os._exit(0)
    mp.spawn(
        train_paradigm,
        args=[world_size, args, parameters],
        nprocs=world_size
    )