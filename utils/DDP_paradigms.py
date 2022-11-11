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

def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '{}'.format(args.port)
    if args.gpu[0] == -1:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

sys.path.append('/root/Cardiac-MRI-Reconstrucion/')

SAVE_INTERVAL = 1

def train_paradigm(rank, world_size, shared_data, args, parameters):
    torch.cuda.set_device(args.gpu[rank])
    setup(rank, world_size, args)
    proc_device = torch.device('cuda:{}'.format(args.gpu[rank]))
    
    if parameters['dataset'] == 'acdc':
        from utils.datasets.ACDC import ACDC as dataset
    if parameters['architecture'] == 'mdcnn':
        from utils.models.MDCNN import MDCNN as Model
        from utils.Trainers.DDP_MDCNNTrainer import Trainer
    trainset = dataset(
                        args.dataset_path, 
                        parameters,
                        train = True, 
                        blank = True,
                    )
    testset = dataset(
                        args.dataset_path, 
                        parameters,
                        train = False, 
                        blank = True,
                    )

    trainset.set_shared_lists(shared_data)
    testset.set_shared_lists(shared_data)

    trainset.rest_init()
    testset.rest_init()

    model = Model(parameters).to(proc_device)
    checkpoint_path = os.path.join(args.run_id, './checkpoints/')

    parameters['GPUs'] = args.gpu
    if rank == 0:
        if args.neptune_log:
            if os.path.isfile(checkpoint_path + 'neptune_run.pth'):
                run_id = torch.load(checkpoint_path + 'neptune_run.pth', map_location = torch.device('cpu'))['run_id']
                run = neptune.init_run(
                    project="fcrl/Cardiac-MRI-Reconstruction",
                    with_id=run_id,
                    name = args.run_id,
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZDU2NDJjMy1lNzczLTRkZDEtODAwYy01MWFlM2VmN2Q4ZTEifQ==",
                )
            else:
                run = neptune.init_run(
                    project="fcrl/Cardiac-MRI-Reconstruction",
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
        # scaler_dict = dic['scaler']
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
        # scaler_dict = None
        scheduler_dict = None
        if rank == 0:
            print('Starting Training', flush = True)

    model = DDP(model, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = False)
    trainer = Trainer(model, trainset, testset, parameters, proc_device, rank, world_size, args)
    if args.resume:
        trainer.optim.load_state_dict(opt_dict)
        # trainer.scaler.load_state_dict(scaler_dict)
        if parameters['scheduler'] != 'None':
            trainer.scheduler.load_state_dict(scheduler_dict)

    for e in range(parameters['num_epochs']):
        if pre_e > 0:
            pre_e -= 1
            continue
        collected_train_losses = [torch.zeros(3,).to(proc_device) for _ in range(world_size)]
        collected_test_losses = [torch.zeros(3,).to(proc_device) for _ in range(world_size)]
        
        lossrecon, lossft, lossreconft = trainer.train(e)
        dist.all_gather(collected_train_losses, torch.tensor([lossrecon, lossft, lossreconft]).to(proc_device))
        if rank == 0:
            avg_train_recon_loss = sum([x[0] for x in collected_train_losses]).item()/len(args.gpu)
            avg_train_ft_loss = sum([x[1] for x in collected_train_losses]).item()/len(args.gpu)
            avg_train_recon_ft_loss = sum([x[2] for x in collected_train_losses]).item()/len(args.gpu)
            if args.neptune_log and rank == 0:
                run["train/train_recon_loss"].log(avg_train_recon_loss)
                run["train/train_ft_loss"].log(avg_train_ft_loss)
                run["train/train_recon_ft_loss"].log(avg_train_recon_ft_loss)
            print('Average Train Recon Loss for Epoch {} = {}' .format(e, avg_train_recon_loss), flush = True)
            if trainer.criterion_FT is not None:
                print('Average Train FT Loss for Epoch {} = {}' .format(e, avg_train_ft_loss), flush = True)
            if trainer.criterion_reconFT is not None:
                print('Average Train Recon FT Loss for Epoch {} = {}' .format(e, avg_train_recon_ft_loss), flush = True)

        test_lossrecon, test_lossft, test_lossreconft,_,_,_ = trainer.evaluate(e, train = False)
        dist.all_gather(collected_test_losses, torch.tensor([test_lossrecon, test_lossft, test_lossreconft]).to(proc_device))
        if rank == 0:
            avg_test_recon_loss = sum([x[0] for x in collected_test_losses]).item()/len(args.gpu)
            avg_test_ft_loss = sum([x[1] for x in collected_test_losses]).item()/len(args.gpu)
            avg_test_recon_ft_loss = sum([x[2] for x in collected_test_losses]).item()/len(args.gpu)
            if args.neptune_log and rank == 0:
                run["train/test_recon_loss"].log(avg_test_recon_loss)
                run["train/test_ft_loss"].log(avg_test_ft_loss)
                run["train/test_recon_ft_loss"].log(avg_test_recon_ft_loss)
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
            # dic['scaler'] = trainer.scaler.state_dict()
            if (e+1) % SAVE_INTERVAL == 0:
                torch.save(dic, checkpoint_path + 'checkpoint_{}.pth'.format(model_state))
                torch.save({'state': model_state}, checkpoint_path + 'state.pth')
                # model_state += 1
                print('Saving model after {} Epochs\n\n'.format(e+1), flush = True)

    cleanup()

def test_paradigm(rank, world_size, shared_data, args, parameters):
    torch.cuda.set_device(args.gpu[rank])
    setup(rank, world_size, args)
    proc_device = torch.device('cuda:{}'.format(args.gpu[rank]))
    if parameters['dataset'] == 'acdc':
        from utils.datasets.ACDC import ACDC as dataset
    if parameters['architecture'] == 'mdcnn':
        from utils.models.MDCNN import MDCNN as Model
        from utils.Trainers.DDP_MDCNNTrainer import Trainer
    trainset = dataset(
                        args.dataset_path, 
                        parameters,
                        train = True, 
                        blank = True,
                    )
    testset = dataset(
                        args.dataset_path, 
                        parameters,
                        train = False, 
                        blank = True,
                    )

    trainset.set_shared_lists(shared_data)
    testset.set_shared_lists(shared_data)

    trainset.rest_init()
    testset.rest_init()

    model = Model(parameters).to(proc_device)
    checkpoint_path = os.path.join(args.run_id, './checkpoints/')

    if rank == 0:
        if args.neptune_log:
            if os.path.isfile(checkpoint_path + 'neptune_run.pth'):
                run_id = torch.load(checkpoint_path + 'neptune_run.pth', map_location = torch.device('cpu'))['run_id']
                run = neptune.init_run(
                    project="fcrl/Cardiac-MRI-Reconstruction",
                    with_id=run_id,
                    name = args.run_id,
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZDU2NDJjMy1lNzczLTRkZDEtODAwYy01MWFlM2VmN2Q4ZTEifQ==",
                )
            else:
                run = neptune.init_run(
                    project="fcrl/Cardiac-MRI-Reconstruction",
                    name = args.run_id,
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZDU2NDJjMy1lNzczLTRkZDEtODAwYy01MWFlM2VmN2Q4ZTEifQ==",
                )
                torch.save({'run_id': run["sys/id"].fetch()}, checkpoint_path + 'neptune_run.pth')
            if not args.resume:
                if run.exists("train"):
                    run["train"].pop()
                if run.exists("test"):
                    run["test"].pop()
                if run.exists("visualize"):
                    run["visualize"].pop()
        else:
            run = None

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
        # scaler_dict = None
        scheduler_dict = None
        if rank == 0:
            print('Starting Training', flush = True)

    model = DDP(model, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = False)
    trainer = Trainer(model, trainset, testset, parameters, proc_device, rank, world_size, args)
    
    if not args.visualise_only:
        test_lossrecon, test_lossft, test_lossreconft, test_ssim, test_l1, test_l2 = trainer.evaluate(pre_e, train = False)
        collected_test_losses = [torch.zeros(6,).to(proc_device) for _ in range(world_size)]
        dist.all_gather(collected_test_losses, torch.tensor([test_lossrecon, test_lossft, test_lossreconft, test_ssim, test_l1, test_l2]).to(proc_device))
        if rank == 0:
            avg_test_recon_loss = sum([x[0] for x in collected_test_losses]).item()/len(args.gpu)
            avg_test_ft_loss = sum([x[1] for x in collected_test_losses]).item()/len(args.gpu)
            avg_test_recon_ft_loss = sum([x[2] for x in collected_test_losses]).item()/len(args.gpu)
            avg_test_ssim = sum([x[3] for x in collected_test_losses]).item()/len(args.gpu)
            avg_test_l1 = sum([x[4] for x in collected_test_losses]).item()/len(args.gpu)
            avg_test_l2 = sum([x[5] for x in collected_test_losses]).item()/len(args.gpu)
            if args.neptune_log and rank == 0:
                run["test/test_recon_loss"] = avg_test_recon_loss
                run["test/test_ft_loss"] = avg_test_ft_loss
                run["test/test_recon_ft_loss"] = avg_test_recon_ft_loss
                run["test/test_ssim_score"] = avg_test_ssim
                run["test/test_l1_loss"] = avg_test_l1
                run["test/test_l2_loss"] = avg_test_l2
            print('Test Loss After {} Epochs:'.format(pre_e), flush = True)
            print('Recon Loss = {}'.format(avg_test_recon_loss), flush = True)
            print('SSIM Score = {}'.format(avg_test_ssim), flush = True)
            print('L1 Loss = {}'.format(avg_test_l1), flush = True)
            print('L2 Loss = {}'.format(avg_test_l2), flush = True)
            if trainer.criterion_FT is not None:
                print('FT Loss = {}'.format(avg_test_ft_loss), flush = True)
            if trainer.criterion_reconFT is not None:
                print('Recon FT Loss = {}'.format(avg_test_recon_ft_loss), flush = True)
        if not args.test_only:
            train_lossrecon, train_lossft, train_lossreconft, train_ssim, train_l1, train_l2 = trainer.evaluate(pre_e, train = True)
            collected_train_losses = [torch.zeros(6,).to(proc_device) for _ in range(world_size)]
            dist.all_gather(collected_train_losses, torch.tensor([train_lossrecon, train_lossft, train_lossreconft, train_ssim, train_l1, train_l2]).to(proc_device))
            if rank == 0:
                avg_train_recon_loss = sum([x[0] for x in collected_train_losses]).item()/len(args.gpu)
                avg_train_ft_loss = sum([x[1] for x in collected_train_losses]).item()/len(args.gpu)
                avg_train_recon_ft_loss = sum([x[2] for x in collected_train_losses]).item()/len(args.gpu)
                avg_train_ssim = sum([x[3] for x in collected_train_losses]).item()/len(args.gpu)
                avg_train_l1 = sum([x[4] for x in collected_train_losses]).item()/len(args.gpu)
                avg_train_l2 = sum([x[5] for x in collected_train_losses]).item()/len(args.gpu)
                if args.neptune_log and rank == 0:
                    run["test/train_recon_loss"] = avg_train_recon_loss
                    run["test/train_ft_loss"] = avg_train_ft_loss
                    run["test/train_recon_ft_loss"] = avg_train_recon_ft_loss
                    run["test/train_ssim_score"] = avg_train_ssim
                    run["test/train_l1_loss"] = avg_train_l1
                    run["test/train_l2_loss"] = avg_train_l2
                print('Train Loss After {} Epochs:'.format(pre_e), flush = True)
                print('Recon Loss = {}'.format(avg_train_recon_loss), flush = True)
                print('SSIM Score = {}'.format(avg_train_ssim), flush = True)
                print('L1 Loss = {}'.format(avg_train_l1), flush = True)
                print('L2 Loss = {}'.format(avg_train_l2), flush = True)
                if trainer.criterion_FT is not None:
                    print('FT Loss = {}'.format(avg_train_ft_loss), flush = True)
                if trainer.criterion_reconFT is not None:
                    print('Recon FT Loss = {}'.format(avg_train_recon_ft_loss), flush = True)
    if rank == 0:
        # if args.neptune_log and rank == 0:
        #     for x in sorted(os.listdir(os.path.join(args.run_id, 'results/train'))):
        #         run['train/{}'.format(x)].upload(File(os.path.join(args.run_id, 'results/train/{}'.format(x))))
        #         break
        #     for x in sorted(os.listdir(os.path.join(args.run_id, 'results/test'))):
        #         run['test/{}'.format(x)].upload(File(os.path.join(args.run_id, 'results/test/{}'.format(x))))
        #         break
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
        plt.savefig(os.path.join(args.run_id, 'results/train_loss.png'))
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
        plt.savefig(os.path.join(args.run_id, 'results/test_loss.png'))

        plt.figure()
        plt.title('Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(losses)), [x[0] for x in losses], label = 'Train Recon Loss: {}'.format(parameters['loss_recon']), color = 'r')
        plt.plot(range(len(test_losses)), [x[0] for x in test_losses], label = 'Test Recon Loss: {}'.format(parameters['loss_recon']), color = 'b')
        if parameters['loss_FT'] != 'None':
            plt.plot(range(len(losses)), [x[1] for x in losses], label = 'Train FT Loss: {}'.format(parameters['loss_FT']), color = 'y')
            plt.plot(range(len(test_losses)), [x[1] for x in test_losses], label = 'Test FT Loss: {}'.format(parameters['loss_FT']), color = 'c')
        if parameters['loss_reconstructed_FT'] != 'None':
            plt.plot(range(len(losses)), [x[2] for x in losses], label = 'Train Recon FT Loss: {}'.format(parameters['loss_reconstructed_FT']), color = 'm')
            plt.plot(range(len(test_losses)), [x[2] for x in test_losses], label = 'Test Recon FT Loss: {}'.format(parameters['loss_reconstructed_FT']), color = 'tab:pink')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(os.path.join(args.run_id, 'results/train_test_loss.png'))
        
        plt.close('all')
        trainer.visualise(pre_e, train = False)
        trainer.visualise(pre_e, train = True)
        with open(os.path.join(args.run_id, 'status.txt'), 'w') as f:
            f.write('1')
    cleanup()