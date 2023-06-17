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
        from utils.myDatasets.ACDC_radial import ACDC_radial as dataset
    from utils.models.periodLSTM import fetch_lstm_type
    Model_Kspace, Model_Ispace = fetch_lstm_type(parameters)
    from utils.Trainers.DDP_LSTMTrainer_nufft import Trainer

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



    kspace_model = Model_Kspace(parameters, proc_device).to(proc_device)
    ispace_model = Model_Ispace(parameters, proc_device).to(proc_device)
    
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

    if args.resume_kspace:
        model_state = 0
        dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state),map_location = proc_device)
        pre_e = dic['e']
        kspace_model.load_state_dict(dic['kspace_model'])
        model_state = 1
        pre_e =0
        losses = []
        test_losses = []
        opt_dict_kspace = None
        opt_dict_ispace = None
        # scaler_dict = None
        scheduler_dict_ispace = None
        scheduler_dict_kspace = None
        if rank == 0:
            print('Loading kspace model after {} epochs'.format(dic['e']), flush = True)
    elif args.resume:
        model_state = torch.load(checkpoint_path + 'state.pth', map_location = torch.device('cpu'))['state']
        if (not args.state == -1):
            model_state = args.state
        if rank == 0:
            print('Loading checkpoint at model state {}'.format(model_state), flush = True)
        dic0 = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(0),map_location = proc_device)
        dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state),map_location = proc_device)
        pre_e = dic['e']
        kspace_model.load_state_dict(dic0['kspace_model'])
        ispace_model.load_state_dict(dic['ispace_model'])
        opt_dict_kspace = dic0['kspace_optim']
        opt_dict_ispace = dic['ispace_optim']
        # scaler_dict = dic['scaler']
        if parameters['scheduler'] != 'None':
            scheduler_dict_ispace = dic['ispace_scheduler']
            scheduler_dict_kspace = dic0['kspace_scheduler']
        losses = dic['losses']
        test_losses = dic['test_losses']
        if rank == 0:
            print('Resuming Training after {} epochs'.format(pre_e), flush = True)
    else:
        model_state = 0
        pre_e =0
        losses = []
        test_losses = []
        opt_dict_kspace = None
        opt_dict_ispace = None
        # scaler_dict = None
        scheduler_dict_ispace = None
        scheduler_dict_kspace = None
        if rank == 0:
            print('Starting Training', flush = True)

    # kspace_model = DDP(kspace_model, device_ids = None, output_device = None, find_unused_parameters = False)
    kspace_model = DDP(kspace_model, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = False)
    ispace_model = DDP(ispace_model, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = False)
    trainer = Trainer(kspace_model, ispace_model, trainset, testset, parameters, proc_device, rank, world_size, args)
    if args.resume:
        trainer.ispace_optim.load_state_dict(opt_dict_ispace)
        trainer.kspace_optim.load_state_dict(opt_dict_kspace)
        # trainer.scaler.load_state_dict(scaler_dict)
        if parameters['scheduler'] != 'None':
            trainer.ispace_scheduler.load_state_dict(scheduler_dict_ispace)
            trainer.kspace_scheduler.load_state_dict(scheduler_dict_kspace)

    for e in range(parameters['num_epochs_ispace'] + parameters['num_epochs_kspace']):
        if pre_e > 0:
            pre_e -= 1
            continue
        collected_train_losses = [torch.zeros(6,).to(proc_device) for _ in range(world_size)]
        collected_test_losses = [torch.zeros(6,).to(proc_device) for _ in range(world_size)]
        
        loss_mag, loss_phase, loss_real, ssim, loss_l1, loss_l2 = trainer.train(e)
        dist.all_gather(collected_train_losses, torch.tensor([loss_mag, loss_phase, loss_real, ssim, loss_l1, loss_l2]).to(proc_device))
        if rank == 0:
            avg_train_mag_loss = sum([x[0] for x in collected_train_losses]).item()/len(args.gpu)
            avg_train_phase_loss = sum([x[1] for x in collected_train_losses]).item()/len(args.gpu)
            avg_train_real_loss = sum([x[2] for x in collected_train_losses]).item()/len(args.gpu)
            ssim_score = sum([x[3] for x in collected_train_losses]).item()/len(args.gpu)
            l1_score = sum([x[3] for x in collected_train_losses]).item()/len(args.gpu)
            l2_score = (sum([x[5] for x in collected_train_losses]).item()/len(args.gpu))**0.5
            if args.neptune_log and rank == 0:
                run["train/train_mag_loss"].log(avg_train_mag_loss)
                run["train/train_phase_loss"].log(avg_train_phase_loss)
                run["train/train_real_loss"].log(avg_train_real_loss)
                run["train/train_ssim"].log(ssim_score)
                run["train/train_l1"].log(l1_score)
                run["train/train_l2"].log(l2_score)
            print('Train Mag Loss for Epoch {} = {}' .format(e, avg_train_mag_loss), flush = True)
            print('Train Phase Loss for Epoch {} = {}' .format(e, avg_train_phase_loss), flush = True)
            print('Train Real Loss for Epoch {} = {}' .format(e, avg_train_real_loss), flush = True)
            print('Train SSIM for Epoch {} = {}' .format(e, ssim_score), flush = True)

        tt = time.time()
        test_loss_mag, test_loss_phase, test_loss_real, test_ssim, test_l1, test_l2 = trainer.evaluate(e, train = False)
        print('Time1',time.time()-tt)
        tt = time.time()
        dist.all_gather(collected_test_losses, torch.tensor([test_loss_mag, test_loss_phase, test_loss_real, test_ssim, test_l1, test_l2]).to(proc_device))
        print('Time2', time.time()-tt)
        if rank == 0:
            avg_test_mag_loss = sum([x[0] for x in collected_test_losses]).item()/len(args.gpu)
            avg_test_phase_loss = sum([x[1] for x in collected_test_losses]).item()/len(args.gpu)
            avg_test_real_loss = sum([x[2] for x in collected_test_losses]).item()/len(args.gpu)
            test_ssim_score = sum([x[3] for x in collected_test_losses]).item()/len(args.gpu)
            test_l1_score = sum([x[4] for x in collected_test_losses]).item()/len(args.gpu)
            test_l2_score = (sum([x[5] for x in collected_test_losses]).item()/len(args.gpu)) ** 0.5
            if args.neptune_log and rank == 0:
                run["train/test_mag_loss"].log(avg_test_mag_loss)
                run["train/test_phase_loss"].log(avg_test_phase_loss)
                run["train/test_real_loss"].log(avg_test_real_loss)
                run["train/test_ssim"].log(test_ssim_score)
                run["train/test_l1"].log(test_l1_score)
                run["train/test_l2"].log(test_l2_score)
            print('Test Mag Loss for Epoch {} = {}' .format(e, avg_test_mag_loss), flush = True)
            print('Test Phase Loss for Epoch {} = {}' .format(e, avg_test_phase_loss), flush = True)
            print('Test Real Loss for Epoch {} = {}' .format(e, avg_test_real_loss), flush = True)
            print('Test SSIM for Epoch {} = {}' .format(e, test_ssim_score), flush = True)

        if rank == 0:
            losses.append((avg_train_mag_loss, avg_train_phase_loss, avg_train_real_loss, ssim_score))
            test_losses.append((avg_test_mag_loss, avg_test_phase_loss, avg_test_real_loss, test_ssim_score))

            parameters['train_losses'] = losses
            parameters['test_losses'] = test_losses

            dic = {}
            dic['e'] = e+1
            dic['kspace_model'] = trainer.kspace_model.module.state_dict()
            dic['ispace_model'] = trainer.ispace_model.module.state_dict()
            dic['kspace_optim'] = trainer.kspace_optim.state_dict()
            dic['ispace_optim'] = trainer.ispace_optim.state_dict()
            if parameters['scheduler'] != 'None':
                dic['kspace_scheduler'] = trainer.kspace_scheduler.state_dict()
                dic['ispace_scheduler'] = trainer.ispace_scheduler.state_dict()
            dic['losses'] = losses
            dic['test_losses'] = test_losses
            # dic['scaler'] = trainer.scaler.state_dict()
            if (e+1) % SAVE_INTERVAL == 0:
                if e > parameters['num_epochs_kspace']:
                    model_state = 1
                torch.save(dic, checkpoint_path + 'checkpoint_{}.pth'.format(model_state))
                torch.save({'state': model_state}, checkpoint_path + 'state.pth')
                # model_state += 1
                print('Saving model after {} Epochs\n\n'.format(e+1), flush = True)

    cleanup()

def test_paradigm(rank, world_size, args, parameters):
    torch.cuda.set_device(args.gpu[rank])
    setup(rank, world_size, args)
    proc_device = torch.device('cuda:{}'.format(args.gpu[rank]))
    if parameters['dataset'] == 'acdc':
        from utils.myDatasets.ACDC_radial import ACDC_radial as dataset
    from utils.models.periodLSTM import fetch_lstm_type
    Model_Kspace, Model_Ispace = fetch_lstm_type(parameters)
    from utils.Trainers.DDP_LSTMTrainer_nufft import Trainer

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



    kspace_model = Model_Kspace(parameters, proc_device).to(proc_device)
    ispace_model = Model_Ispace(parameters, proc_device).to(proc_device)
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

    if args.resume_kspace:
        model_state = 0
        dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state),map_location = proc_device)
        pre_e = dic['e']
        kspace_model.load_state_dict(dic['kspace_model'])
        model_state = 1
        pre_e =0
        losses = []
        test_losses = []
        opt_dict_kspace = None
        opt_dict_ispace = None
        # scaler_dict = None
        scheduler_dict_ispace = None
        scheduler_dict_kspace = None
        if rank == 0:
            print('Loading kspace model after {} epochs'.format(dic['e']), flush = True)
    elif args.resume:
        model_state = torch.load(checkpoint_path + 'state.pth', map_location = torch.device('cpu'))['state']
        if (not args.state == -1):
            model_state = args.state
        if rank == 0:
            print('Loading checkpoint at model state {}'.format(model_state), flush = True)
        dic0 = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(0),map_location = proc_device)
        dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state),map_location = proc_device)
        pre_e = dic['e']
        kspace_model.load_state_dict(dic0['kspace_model'])
        ispace_model.load_state_dict(dic['ispace_model'])
        opt_dict_kspace = dic0['kspace_optim']
        opt_dict_ispace = dic['ispace_optim']
        # scaler_dict = dic['scaler']
        if rank == 0:
            print('Loading kspace model after {} epochs'.format(dic0['e']), flush = True)
        if parameters['scheduler'] != 'None':
            scheduler_dict_ispace = dic['ispace_scheduler']
            scheduler_dict_kspace = dic0['kspace_scheduler']
        losses = dic['losses']
        test_losses = dic['test_losses']
        if rank == 0:
            print('Resuming Training after {} epochs'.format(pre_e), flush = True)
    else:
        model_state = 0
        pre_e =0
        losses = []
        test_losses = []
        # scaler_dict = None
        scheduler_dict = None
        if rank == 0:
            print('Starting Training', flush = True)

    # kspace_model = DDP(kspace_model, device_ids = None, output_device = None, find_unused_parameters = False)
    kspace_model = DDP(kspace_model, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = False)
    ispace_model = DDP(ispace_model, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = False)
    trainer = Trainer(kspace_model, ispace_model, trainset, testset, parameters, proc_device, rank, world_size, args)
    
    if rank == 0:
        # if args.neptune_log and rank == 0:
        #     for x in sorted(os.listdir(os.path.join(args.run_id, 'images/train'))):
        #         run['train/{}'.format(x)].upload(File(os.path.join(args.run_id, 'images/train/{}'.format(x))))
        #         break
        #     for x in sorted(os.listdir(os.path.join(args.run_id, 'images/test'))):
        #         run['test/{}'.format(x)].upload(File(os.path.join(args.run_id, 'images/test/{}'.format(x))))
        #         break
        plt.figure()
        plt.title('Train Mag Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(losses)), [x[0] for x in losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(args.run_id, 'images/train_mag_loss.png'))
        plt.figure()
        plt.title('Train Phase Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(losses)), [x[1] for x in losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(args.run_id, 'images/train_phase_loss.png'))
        plt.figure()
        plt.title('Train Real Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(losses)), [x[2] for x in losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(args.run_id, 'images/train_real_loss.png'))
        plt.figure()
        plt.title('Train SSIM after {} epochs'.format(pre_e))
        plt.plot(range(len(losses)), [x[3] for x in losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('ssim')
        plt.savefig(os.path.join(args.run_id, 'images/train_ssim.png'))

        plt.figure()
        plt.title('Test Mag Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(test_losses)), [x[0] for x in test_losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(args.run_id, 'images/test_mag_loss.png'))
        plt.figure()
        plt.title('Test Phase Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(test_losses)), [x[1] for x in test_losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(args.run_id, 'images/test_phase_loss.png'))
        plt.figure()
        plt.title('Test Real Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(test_losses)), [x[2] for x in test_losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(args.run_id, 'images/test_real_loss.png'))
        plt.figure()
        plt.title('Test SSIM after {} epochs'.format(pre_e))
        plt.plot(range(len(test_losses)), [x[3] for x in test_losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('ssim')
        plt.savefig(os.path.join(args.run_id, 'images/test_ssim.png'))
        
        plt.close('all')
        trainer.visualise(pre_e, train = False)
        trainer.visualise(pre_e, train = True)

    if not args.visualise_only:
        test_loss_mag, test_loss_phase, test_loss_real, test_ssim, test_loss_l1, test_loss_l2 = trainer.evaluate(pre_e, train = False)
        collected_test_losses = [torch.zeros(6,).to(proc_device) for _ in range(world_size)]
        dist.all_gather(collected_test_losses, torch.tensor([test_loss_mag, test_loss_phase, test_loss_real, test_ssim, test_loss_l1, test_loss_l2]).to(proc_device))
        if rank == 0:
            avg_test_mag_loss = sum([x[0] for x in collected_test_losses]).item()/len(args.gpu)
            avg_test_phase_loss = sum([x[1] for x in collected_test_losses]).item()/len(args.gpu)
            avg_test_real_loss = sum([x[2] for x in collected_test_losses]).item()/len(args.gpu)
            avg_test_ssim = sum([x[3] for x in collected_test_losses]).item()/len(args.gpu)
            avg_test_l1 = sum([x[4] for x in collected_test_losses]).item()/len(args.gpu)
            avg_test_l2 = (sum([x[5] for x in collected_test_losses]).item()/len(args.gpu))**0.5
            if args.neptune_log and rank == 0:
                run["test/test_mag_loss"] = avg_test_mag_loss
                run["test/test_phase_loss"] = avg_test_phase_loss
                run["test/test_real_loss"] = avg_test_real_loss
                run["test/test_ssim_score"] = avg_test_ssim
                run["test/test_l1_loss"] = avg_test_l1
                run["test/test_l2_loss"] = avg_test_l2
            print('Test Mag Loss = {}'.format(avg_test_mag_loss), flush = True)
            print('Test Phase Loss = {}'.format(avg_test_phase_loss), flush = True)
            print('Test Real Loss = {}'.format(avg_test_real_loss), flush = True)
            print('Test SSIM Score = {}'.format(avg_test_ssim), flush = True)
            print('Test L1 Loss = {}'.format(avg_test_l1), flush = True)
            print('Test L2 Loss = {}'.format(avg_test_l2), flush = True)

        if not args.test_only:
            train_loss_mag, train_loss_phase, train_loss_real, train_ssim, train_loss_l1, train_loss_l2 = trainer.evaluate(pre_e, train = True)
            collected_train_losses = [torch.zeros(6,).to(proc_device) for _ in range(world_size)]
            dist.all_gather(collected_train_losses, torch.tensor([train_loss_mag, train_loss_phase, train_loss_real, train_ssim, train_loss_l1, train_loss_l2]).to(proc_device))
            if rank == 0:
                avg_train_mag_loss = sum([x[0] for x in collected_train_losses]).item()/len(args.gpu)
                avg_train_phase_loss = sum([x[1] for x in collected_train_losses]).item()/len(args.gpu)
                avg_train_real_loss = sum([x[2] for x in collected_train_losses]).item()/len(args.gpu)
                avg_train_ssim = sum([x[3] for x in collected_train_losses]).item()/len(args.gpu)
                avg_train_l1 = sum([x[4] for x in collected_train_losses]).item()/len(args.gpu)
                avg_train_l2 = (sum([x[5] for x in collected_train_losses]).item()/len(args.gpu))**0.5
                if args.neptune_log and rank == 0:
                    run["test/train_mag_loss"] = avg_train_mag_loss
                    run["test/train_phase_loss"] = avg_train_phase_loss
                    run["test/train_real_loss"] = avg_train_real_loss
                    run["test/train_ssim_score"] = avg_train_ssim
                    run["test/train_l1_loss"] = avg_train_l1
                    run["test/train_l2_loss"] = avg_train_l2
                print('Train Mag Loss = {}'.format(avg_train_mag_loss), flush = True)
                print('Train Phase Loss = {}'.format(avg_train_phase_loss), flush = True)
                print('Train Real Loss = {}'.format(avg_train_real_loss), flush = True)
                print('Train SSIM Score = {}'.format(avg_train_ssim), flush = True)
                print('Train L1 Loss = {}'.format(avg_train_l1), flush = True)
                print('Train L2 Loss = {}'.format(avg_train_l2), flush = True)

    if rank == 0:
        with open(os.path.join(args.run_id, 'status.txt'), 'w') as f:
            f.write('1')

    cleanup()