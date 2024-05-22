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

def train_paradigm(rank, world_size, args, parameters):
    torch.cuda.set_device(args.gpu[rank])
    setup(rank, world_size, args)
    proc_device = torch.device('cuda:{}'.format(args.gpu[rank]))
    
    if parameters['dataset'] == 'acdc':
        from utils.myDatasets.ACDC_radial_faster import ACDC_radial as dataset
    from utils.models.MDCNN import MDCNN
    model = MDCNN(parameters, proc_device).to(args.gpu[rank])
    from utils.Trainers.DDP_MDCNNTrainer_nufft import Trainer

    trainset = dataset(
                        args.dataset_path, 
                        parameters,
                        proc_device,
                        train = True, 
                    )
    testset = dataset(
                        args.dataset_path, 
                        parameters,
                        proc_device,
                        train = False, 
                    )


    checkpoint_path = os.path.join(args.run_id, 'checkpoints/')

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

        if parameters['scheduler'] != 'None':
            scheduler_dict = dic['scheduler']
        losses = dic['losses']
        test_losses = dic['test_losses']
        if rank == 0:
            print('Resuming Training after {} epochs'.format(pre_e), flush = True)
        del dic
    else:
        model_state = 0
        pre_e =0
        losses = []
        test_losses = []
        opt_dict = None
        scheduler_dict = None
        if rank == 0:
            print('Starting Training', flush = True)

    model = DDP(model, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = False)
    trainer = Trainer(model, trainset, testset, parameters, proc_device, rank, world_size, args)
    if args.time_analysis:
        if rank == 0:
            trainer.time_analysis()
        return

    
    if args.resume:
        trainer.optim.load_state_dict(opt_dict)
        if parameters['scheduler'] != 'None':
            trainer.scheduler.load_state_dict(scheduler_dict)

    for e in range(parameters['num_epochs']):
        if pre_e > 0:
            pre_e -= 1
            continue
        collected_train_losses = [torch.zeros(4,).to(proc_device) for _ in range(world_size)]
        collected_test_losses = [torch.zeros(4,).to(proc_device) for _ in range(world_size)]
        
        ispaceloss, ispacessim, ipsaceloss_l1, ispaceloss_l2 = trainer.train(e)
        dist.all_gather(collected_train_losses, torch.tensor([ispaceloss, ispacessim, ipsaceloss_l1, ispaceloss_l2]).to(proc_device))
        if rank == 0:
            avgispace_train_loss = sum([x[0] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            ispacessim_score = sum([x[1] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            ispacel1_score = sum([x[2] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            ispacel2_score = (sum([x[3] for x in collected_train_losses]).cpu().item()/len(args.gpu))**0.5
            if args.neptune_log and rank == 0:
                run["train/ispace_train_loss"].log(avgispace_train_loss)
                run["train/ispace_train_ssim"].log(ispacessim_score)
                run["train/ispace_train_l1"].log(ispacel1_score)
                run["train/ispace_train_l2"].log(ispacel2_score)
            
            print('ISpace Training Losses:', flush = True)
            print('ISpace Real (L1) Loss = {}' .format(avgispace_train_loss), flush = True)
            print('ISpace SSIM = {}\n\n' .format(ispacessim_score), flush = True)

        tt = time.time()
        ispaceloss, ispacessim, ipsaceloss_l1, ispaceloss_l2 = trainer.evaluate(e, train = False)
        print('Time1',time.time()-tt)
        tt = time.time()
        dist.all_gather(collected_test_losses, torch.tensor([ispaceloss, ispacessim, ipsaceloss_l1, ispaceloss_l2]).to(proc_device))
        print('Time2', time.time()-tt)
        if rank == 0:
            avgispace_test_loss = sum([x[0] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_ispacessim_score = sum([x[1] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_ispacel1_score = sum([x[2] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_ispacel2_score = (sum([x[3] for x in collected_test_losses]).cpu().item()/len(args.gpu)) ** 0.5
            if args.neptune_log and rank == 0:
                run["train/ispacetest_loss"].log(avgispace_test_loss)
                run["train/ispacetest_ssim"].log(test_ispacessim_score)
                run["train/ispacetest_l1"].log(test_ispacel1_score)
                run["train/ispacetest_l2"].log(test_ispacel2_score)
            
            print('ISpace Training Losses:', flush = True)
            print('ISpace Real (L1) Loss = {}' .format(avgispace_test_loss), flush = True)
            print('ISpace SSIM = {}' .format(test_ispacessim_score), flush = True)
        if rank == 0:
            losses.append((avgispace_train_loss, ispacessim_score))
            test_losses.append((avgispace_test_loss, test_ispacessim_score))

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
            if (e+1) % SAVE_INTERVAL == 0:
                torch.save(dic, checkpoint_path + 'checkpoint_{}.pth'.format(model_state))
                torch.save({'state': model_state}, checkpoint_path + 'state.pth')
                print('Saving model after {} Epochs\n\n'.format(e+1), flush = True)
            del dic
        del collected_test_losses
        del collected_train_losses
        torch.cuda.empty_cache()

    cleanup()

def test_paradigm(rank, world_size, args, parameters):
    torch.cuda.set_device(args.gpu[rank])
    setup(rank, world_size, args)
    proc_device = torch.device('cuda:{}'.format(args.gpu[rank]))
    
    if parameters['dataset'] == 'acdc':
        from utils.myDatasets.ACDC_radial_faster import ACDC_radial as dataset
    from utils.models.MDCNN import MDCNN
    model = MDCNN(parameters, proc_device).to(args.gpu[rank])
    from utils.Trainers.DDP_MDCNNTrainer_nufft import Trainer

    trainset = dataset(
                        args.dataset_path, 
                        parameters,
                        proc_device,
                        train = True, 
                        visualise_only = args.visualise_only 
                    )
    testset = dataset(
                        args.dataset_path, 
                        parameters,
                        proc_device,
                        train = False,
                        visualise_only = args.visualise_only 
                    )



    checkpoint_path = os.path.join(args.run_id, 'checkpoints/')

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
        dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state),map_location = proc_device)
        pre_e = dic['e']
        model.load_state_dict(dic['model'])
        opt_dict = dic['optim']

        if parameters['scheduler'] != 'None':
            scheduler_dict = dic['scheduler']
        losses = dic['losses']
        test_losses = dic['test_losses']
        if rank == 0:
            print('Resuming Training after {} epochs'.format(pre_e), flush = True)
        del dic
    else:
        model_state = 0
        pre_e =0
        losses = []
        test_losses = []
        opt_dict = None
        scheduler_dict = None

    model = DDP(model, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = False)
    trainer = Trainer(model, trainset, testset, parameters, proc_device, rank, world_size, args)

    if args.time_analysis:
        if rank == 0:
            trainer.time_analysis()
        return
    
    if rank == 0:
        # if args.neptune_log and rank == 0:
        #     for x in sorted(os.listdir(os.path.join(args.run_id, 'images/train'))):
        #         run['train/{}'.format(x)].upload(File(os.path.join(args.run_id, 'images/train/{}'.format(x))))
        #         break
        #     for x in sorted(os.listdir(os.path.join(args.run_id, 'images/test'))):
        #         run['test/{}'.format(x)].upload(File(os.path.join(args.run_id, 'images/test/{}'.format(x))))
        #         break
        plt.figure()
        plt.title('Train Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(losses)), [x[0] for x in losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(args.run_id, 'images/train_loss.png'))
        plt.figure()
        plt.title('Train SSIM after {} epochs'.format(pre_e))
        plt.plot(range(len(losses)), [x[1] for x in losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('ssim')
        plt.savefig(os.path.join(args.run_id, 'images/train_ssim.png'))
        plt.figure()

        plt.figure()
        plt.title('Test Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(test_losses)), [x[0] for x in test_losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(args.run_id, 'images/test_loss.png'))
        plt.figure()
        plt.title('Test SSIM after {} epochs'.format(pre_e))
        plt.plot(range(len(test_losses)), [x[1] for x in test_losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('ssim')
        plt.savefig(os.path.join(args.run_id, 'images/test_ssim.png'))
        
        plt.close('all')
        if not args.numbers_only:
            trainer.visualise(pre_e, train = False)
            trainer.visualise(pre_e, train = True)

    if not args.visualise_only:
        test_ispaceloss, test_ispacessim, test_ispaceloss_l1, test_ispaceloss_l2 = trainer.evaluate(pre_e, train = False)
        collected_test_losses = [torch.zeros(4,).to(proc_device) for _ in range(world_size)]
        dist.all_gather(collected_test_losses, torch.tensor([test_ispaceloss, test_ispacessim, test_ispaceloss_l1, test_ispaceloss_l2]).to(proc_device))
        if rank == 0:
            avgispace_test_loss = sum([x[0] for x in collected_test_losses]).item()/len(args.gpu)
            avgispace_test_ssim = sum([x[1] for x in collected_test_losses]).item()/len(args.gpu)
            avgispace_test_l1 = sum([x[2] for x in collected_test_losses]).item()/len(args.gpu)
            avgispace_test_l2 = (sum([x[3] for x in collected_test_losses]).item()/len(args.gpu))**0.5
            if args.neptune_log and rank == 0:
                run["test/ispacetest_loss"] = avgispace_test_loss
                run["test/ispacetest_ssim_score"] = avgispace_test_ssim
                run["test/ispacetest_l1_loss"] = avgispace_test_l1
                run["test/ispacetest_l2_loss"] = avgispace_test_l2

            print('ISpace Test Losses:', flush = True)
            print('ISpace Real (L1) Loss = {}' .format(avgispace_test_loss), flush = True)
            print('ISpace SSIM = {}\n\n' .format(avgispace_test_ssim), flush = True)

        if not args.test_only:
            train_ispaceloss, train_ispacessim, train_ispaceloss_l1, train_ispaceloss_l2 = trainer.evaluate(pre_e, train = True)
            collected_train_losses = [torch.zeros(4,).to(proc_device) for _ in range(world_size)]
            dist.all_gather(collected_train_losses, torch.tensor([train_ispaceloss, train_ispacessim, train_ispaceloss_l1, train_ispaceloss_l2]).to(proc_device))
            if rank == 0:
                avgispace_train_loss = sum([x[0] for x in collected_train_losses]).item()/len(args.gpu)
                avgispace_train_ssim = sum([x[1] for x in collected_train_losses]).item()/len(args.gpu)
                avgispace_train_l1 = sum([x[2] for x in collected_train_losses]).item()/len(args.gpu)
                avgispace_train_l2 = (sum([x[3] for x in collected_train_losses]).item()/len(args.gpu))**0.5
                if args.neptune_log and rank == 0:
                    run["test/ispacetrain_loss"] = avgispace_train_loss
                    run["test/ispacetrain_ssim_score"] = avgispace_train_ssim
                    run["test/ispacetrain_l1_loss"] = avgispace_train_l1
                    run["test/ispacetrain_l2_loss"] = avgispace_train_l2

                print('ISpace Test Losses:', flush = True)
                print('ISpace Real (L1) Loss = {}' .format(avgispace_train_loss), flush = True)
                print('ISpace SSIM = {}\n\n' .format(avgispace_train_ssim), flush = True)

    if rank == 0:
        with open(os.path.join(args.run_id, 'status.txt'), 'w') as f:
            f.write('1')

    cleanup()