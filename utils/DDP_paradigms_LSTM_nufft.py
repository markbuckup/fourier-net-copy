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
    if 'LSTM' in parameters['kspace_architecture']:
        from utils.models.periodLSTM import fetch_lstm_type as rnn_func
    elif 'GRU' in parameters['kspace_architecture']:
        from utils.models.periodGRU import fetch_gru_type as rnn_func
    Model_Kspace, Model_Ispace = rnn_func(parameters)
    from utils.Trainers.DDP_LSTMTrainer_nufft import Trainer

    temp = os.getcwd().split('/')
    temp = temp[temp.index('experiments'):]
    save_path = os.path.join(parameters['save_folder'], '/'.join(temp))
    save_path = os.path.join(save_path, args.run_id)

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
    
    checkpoint_path = os.path.join(save_path, 'checkpoints/')
    os.makedirs(checkpoint_path, exist_ok = True)

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
        kspace_model.load_state_dict(dic['kspace_model'])
        ispace_model.load_state_dict(dic['ispace_model'])
        if parameters['kspace_architecture'] == 'KLSTM1':
            opt_dict_kspace_mag = dic['kspace_optim_mag']
            opt_dict_kspace_phase = dic['kspace_optim_phase']
        elif parameters['kspace_architecture'] == 'KLSTM2':
            opt_dict_kspace = dic['kspace_optim']
        opt_dict_ispace = dic['ispace_optim']
        # scaler_dict = dic['scaler']
        if parameters['scheduler'] != 'None':
            scheduler_dict_ispace = dic['ispace_scheduler']
            if parameters['kspace_architecture'] == 'KLSTM1':
                scheduler_dict_kspace_phase = dic['kspace_scheduler_phase']
                scheduler_dict_kspace_mag = dic['kspace_scheduler_mag']
            elif parameters['kspace_architecture'] == 'KLSTM2':
                scheduler_dict_kspace = dic['kspace_scheduler']
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

    if args.time_analysis:
        if rank == 0:
            trainer.time_analysis()
        return

    if args.resume:
        trainer.ispace_optim.load_state_dict(opt_dict_ispace)
        if parameters['kspace_architecture'] == 'KLSTM1':
            trainer.kspace_optim_mag.load_state_dict(opt_dict_kspace_mag)
            trainer.kspace_optim_phase.load_state_dict(opt_dict_kspace_phase)
        elif parameters['kspace_architecture'] == 'KLSTM2':
            trainer.kspace_optim.load_state_dict(opt_dict_kspace)
        # trainer.scaler.load_state_dict(scaler_dict)
        if parameters['scheduler'] != 'None':
            trainer.ispace_scheduler.load_state_dict(scheduler_dict_ispace)
            if parameters['kspace_architecture'] == 'KLSTM1':
                trainer.kspace_scheduler_phase.load_state_dict(scheduler_dict_kspace_phase)
                trainer.kspace_scheduler_mag.load_state_dict(scheduler_dict_kspace_mag)
            elif parameters['kspace_architecture'] == 'KLSTM2':
                trainer.kspace_scheduler.load_state_dict(scheduler_dict_kspace)

    for e in range(parameters['num_epochs_ispace'] + parameters['num_epochs_kspace']):
        if pre_e > 0:
            pre_e -= 1
            continue
        collected_train_losses = [torch.zeros(10,).to(proc_device) for _ in range(world_size)]
        collected_test_losses = [torch.zeros(10,).to(proc_device) for _ in range(world_size)]
        
        kspaceloss_mag, kpsaceloss_phase, kspaceloss_real, kspacessim, kpsaceloss_l1, kspaceloss_l2, ispaceloss_real, ispacessim, ipsaceloss_l1, ispaceloss_l2 = trainer.train(e)
        dist.all_gather(collected_train_losses, torch.tensor([kspaceloss_mag, kpsaceloss_phase, kspaceloss_real, kspacessim, kpsaceloss_l1, kspaceloss_l2, ispaceloss_real, ispacessim, ipsaceloss_l1, ispaceloss_l2]).to(proc_device))
        if rank == 0:
            avgkspace_train_mag_loss = sum([x[0] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            avgkspace_train_phase_loss = sum([x[1] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            avgkspace_train_real_loss = sum([x[2] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            kspacessim_score = sum([x[3] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            kspacel1_score = sum([x[3] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            kspacel2_score = (sum([x[5] for x in collected_train_losses]).cpu().item()/len(args.gpu))**0.5
            avgispace_train_real_loss = sum([x[6] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            ispacessim_score = sum([x[7] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            ispacel1_score = sum([x[8] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            ispacel2_score = (sum([x[9] for x in collected_train_losses]).cpu().item()/len(args.gpu))**0.5
            if args.neptune_log and rank == 0:
                run["train/kspace_train_mag_loss"].log(avgkspace_train_mag_loss)
                run["train/kspace_train_phase_loss"].log(avgkspace_train_phase_loss)
                run["train/kspace_train_real_loss"].log(avgkspace_train_real_loss)
                run["train/kspace_train_ssim"].log(kspacessim_score)
                run["train/kspace_train_l1"].log(kspacel1_score)
                run["train/kspace_train_l2"].log(kspacel2_score)
                run["train/ispace_train_real_loss"].log(avgispace_train_real_loss)
                run["train/ispace_train_ssim"].log(ispacessim_score)
                run["train/ispace_train_l1"].log(ispacel1_score)
                run["train/ispace_train_l2"].log(ispacel2_score)
            
            print('KSpace Training Losses for Epoch {}:'.format(e), flush = True)
            print('KSpace Mag Loss = {}' .format(avgkspace_train_mag_loss), flush = True)
            print('KSpace Phase Loss = {}' .format(avgkspace_train_phase_loss), flush = True)
            print('KSpace Real Loss = {}' .format(avgkspace_train_real_loss), flush = True)
            print('KSpace SSIM = {}' .format(kspacessim_score), flush = True)
            print('ISpace Training Losses:', flush = True)
            print('ISpace Real (L1) Loss = {}' .format(avgispace_train_real_loss), flush = True)
            print('ISpace SSIM = {}\n\n' .format(ispacessim_score), flush = True)

        tt = time.time()
        kspaceloss_mag, kpsaceloss_phase, kspaceloss_real, kspacessim, kpsaceloss_l1, kspaceloss_l2, ispaceloss_real, ispacessim, ipsaceloss_l1, ispaceloss_l2 = trainer.evaluate(e, train = False)
        # print('Time1',time.time()-tt)
        tt = time.time()
        dist.all_gather(collected_test_losses, torch.tensor([kspaceloss_mag, kpsaceloss_phase, kspaceloss_real, kspacessim, kpsaceloss_l1, kspaceloss_l2, ispaceloss_real, ispacessim, ipsaceloss_l1, ispaceloss_l2]).to(proc_device))
        # print('Time2', time.time()-tt)
        if rank == 0:
            avgkspace_test_mag_loss = sum([x[0] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            avgkspace_test_phase_loss = sum([x[1] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            avgkspace_test_real_loss = sum([x[2] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_kspacessim_score = sum([x[3] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_kspacel1_score = sum([x[4] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_kspacel2_score = (sum([x[5] for x in collected_test_losses]).cpu().item()/len(args.gpu)) ** 0.5
            avgispace_test_real_loss = sum([x[6] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_ispacessim_score = sum([x[7] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_ispacel1_score = sum([x[8] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_ispacel2_score = (sum([x[9] for x in collected_test_losses]).cpu().item()/len(args.gpu)) ** 0.5
            if args.neptune_log and rank == 0:
                run["train/kspacetest_mag_loss"].log(avgkspace_test_mag_loss)
                run["train/kspacetest_phase_loss"].log(avgkspace_test_phase_loss)
                run["train/kspacetest_real_loss"].log(avgkspace_test_real_loss)
                run["train/kspacetest_ssim"].log(test_kspacessim_score)
                run["train/kspacetest_l1"].log(test_kspacel1_score)
                run["train/kspacetest_l2"].log(test_kspacel2_score)
                run["train/ispacetest_real_loss"].log(avgispace_test_real_loss)
                run["train/ispacetest_ssim"].log(test_ispacessim_score)
                run["train/ispacetest_l1"].log(test_ispacel1_score)
                run["train/ispacetest_l2"].log(test_ispacel2_score)
            
            print('KSpace Training Losses for Epoch {}:'.format(e), flush = True)
            print('KSpace Mag Loss = {}' .format(avgkspace_test_mag_loss), flush = True)
            print('KSpace Phase Loss = {}' .format(avgkspace_test_phase_loss), flush = True)
            print('KSpace Real Loss = {}' .format(avgkspace_test_real_loss), flush = True)
            print('KSpace SSIM = {}' .format(test_kspacessim_score), flush = True)
            print('ISpace Training Losses:', flush = True)
            print('ISpace Real (L1) Loss = {}' .format(avgispace_test_real_loss), flush = True)
            print('ISpace SSIM = {}' .format(test_ispacessim_score), flush = True)
        if rank == 0:
            losses.append((avgkspace_train_mag_loss, avgkspace_train_phase_loss, avgkspace_train_real_loss, kspacessim_score, avgispace_train_real_loss, ispacessim_score))
            test_losses.append((avgkspace_test_mag_loss, avgkspace_test_phase_loss, avgkspace_test_real_loss, test_kspacessim_score, avgispace_test_real_loss, test_ispacessim_score))

            parameters['train_losses'] = losses
            parameters['test_losses'] = test_losses

            dic = {}
            dic['e'] = e+1
            dic['kspace_model'] = trainer.kspace_model.module.state_dict()
            dic['ispace_model'] = trainer.ispace_model.module.state_dict()
            if parameters['kspace_architecture'] == 'KLSTM1':
                dic['kspace_optim_mag'] = trainer.kspace_optim_mag.state_dict()
                dic['kspace_optim_phase'] = trainer.kspace_optim_phase.state_dict()
            elif parameters['kspace_architecture'] == 'KLSTM2':
                dic['kspace_optim'] = trainer.kspace_optim.state_dict()

            dic['ispace_optim'] = trainer.ispace_optim.state_dict()
            if parameters['scheduler'] != 'None':
                dic['ispace_scheduler'] = trainer.ispace_scheduler.state_dict()
                if parameters['kspace_architecture'] == 'KLSTM1':
                    dic['kspace_scheduler_phase'] = trainer.kspace_scheduler_phase.state_dict()
                    dic['kspace_scheduler_mag'] = trainer.kspace_scheduler_mag.state_dict()
                elif parameters['kspace_architecture'] == 'KLSTM2':
                    dic['kspace_scheduler'] = trainer.kspace_scheduler.state_dict()
            dic['losses'] = losses
            dic['test_losses'] = test_losses
            # dic['scaler'] = trainer.scaler.state_dict()
            if (e+1) % SAVE_INTERVAL == 0:
                # if e > parameters['num_epochs_kspace']:
                #     model_state = 1
                torch.save(dic, checkpoint_path + 'checkpoint_{}.pth'.format(model_state))
                torch.save({'state': model_state}, checkpoint_path + 'state.pth')
                # model_state += 1
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
    if 'LSTM' in parameters['kspace_architecture']:
        from utils.models.periodLSTM import fetch_lstm_type as rnn_func
    elif 'GRU' in parameters['kspace_architecture']:
        from utils.models.periodGRU import fetch_gru_type as rnn_func
    Model_Kspace, Model_Ispace = rnn_func(parameters)
    from utils.Trainers.DDP_LSTMTrainer_nufft import Trainer

    temp = os.getcwd().split('/')
    temp = temp[temp.index('experiments'):]
    save_path = os.path.join(parameters['save_folder'], '/'.join(temp))
    save_path = os.path.join(save_path, args.run_id)

    trainset = dataset(
                        args.dataset_path, 
                        parameters,
                        proc_device,
                        train = True, 
                        visualise_only = args.visualise_only or args.numbers_only 
                    )
    testset = dataset(
                        args.dataset_path, 
                        parameters,
                        proc_device,
                        train = False,
                        visualise_only = args.visualise_only or args.numbers_only 
                    )



    kspace_model = Model_Kspace(parameters, proc_device).to(proc_device)
    ispace_model = Model_Ispace(parameters, proc_device).to(proc_device)
    checkpoint_path = os.path.join(save_path, 'checkpoints/')

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
        kspace_model.load_state_dict(dic['kspace_model'])
        ispace_model.load_state_dict(dic['ispace_model'])
        # scaler_dict = dic['scaler']
        if rank == 0:
            print('Loading kspace model after {} epochs'.format(dic['e']), flush = True)
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
        # scaler_dict = None
        scheduler_dict = None
        if rank == 0:
            print('Starting Training', flush = True)

    # kspace_model = DDP(kspace_model, device_ids = None, output_device = None, find_unused_parameters = False)
    kspace_model = DDP(kspace_model, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = False)
    ispace_model = DDP(ispace_model, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = False)
    trainer = Trainer(kspace_model, ispace_model, trainset, testset, parameters, proc_device, rank, world_size, args)

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
        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        plt.figure()
        plt.title('Train Mag Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(losses)), [x[0] for x in losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/kspace_train_mag_loss.png'))
        plt.figure()
        plt.title('Train Phase Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(losses)), [x[1] for x in losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/kspace_train_phase_loss.png'))
        plt.figure()
        plt.title('Train Real Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(losses)), [x[2] for x in losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/kspace_train_real_loss.png'))
        plt.figure()
        plt.title('Train SSIM after {} epochs'.format(pre_e))
        plt.plot(range(len(losses)), [x[3] for x in losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('ssim')
        plt.savefig(os.path.join(save_path, 'images/kspace_train_ssim.png'))
        plt.figure()
        plt.title('Train Real Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(losses)), [x[4] for x in losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/ispace_train_real_loss.png'))
        plt.figure()
        plt.title('Train SSIM after {} epochs'.format(pre_e))
        plt.plot(range(len(losses)), [x[5] for x in losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('ssim')
        plt.savefig(os.path.join(save_path, 'images/ispace_train_ssim.png'))

        plt.figure()
        plt.title('Test Mag Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(test_losses)), [x[0] for x in test_losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/kspace_test_mag_loss.png'))
        plt.figure()
        plt.title('Test Phase Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(test_losses)), [x[1] for x in test_losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/kspace_test_phase_loss.png'))
        plt.figure()
        plt.title('Test Real Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(test_losses)), [x[2] for x in test_losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/kspace_test_real_loss.png'))
        plt.figure()
        plt.title('Test SSIM after {} epochs'.format(pre_e))
        plt.plot(range(len(test_losses)), [x[3] for x in test_losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('ssim')
        plt.savefig(os.path.join(save_path, 'images/kspace_test_ssim.png'))
        plt.figure()
        plt.title('Test Real Loss after {} epochs'.format(pre_e))
        plt.plot(range(len(test_losses)), [x[4] for x in test_losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/ispace_test_real_loss.png'))
        plt.figure()
        plt.title('Test SSIM after {} epochs'.format(pre_e))
        plt.plot(range(len(test_losses)), [x[5] for x in test_losses], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('ssim')
        plt.savefig(os.path.join(save_path, 'images/ispace_test_ssim.png'))
        
        plt.close('all')
        if not args.numbers_only:
            trainer.visualise(pre_e, train = False)
            trainer.visualise(pre_e, train = True)

    if not args.visualise_only:
        test_kspaceloss_mag, test_kspaceloss_phase, test_kspaceloss_real, test_kspacessim, test_kspaceloss_l1, test_kspaceloss_l2, test_ispaceloss_real, test_ispacessim, test_ispaceloss_l1, test_ispaceloss_l2 = trainer.evaluate(pre_e, train = False)
        collected_test_losses = [torch.zeros(10,).to(proc_device) for _ in range(world_size)]
        dist.all_gather(collected_test_losses, torch.tensor([test_kspaceloss_mag, test_kspaceloss_phase, test_kspaceloss_real, test_kspacessim, test_kspaceloss_l1, test_kspaceloss_l2, test_ispaceloss_real, test_ispacessim, test_ispaceloss_l1, test_ispaceloss_l2]).to(proc_device))
        if rank == 0:
            avgkspace_test_mag_loss = sum([x[0] for x in collected_test_losses]).item()/len(args.gpu)
            avgkspace_test_phase_loss = sum([x[1] for x in collected_test_losses]).item()/len(args.gpu)
            avgkspace_test_real_loss = sum([x[2] for x in collected_test_losses]).item()/len(args.gpu)
            avgkspace_test_ssim = sum([x[3] for x in collected_test_losses]).item()/len(args.gpu)
            avgkspace_test_l1 = sum([x[4] for x in collected_test_losses]).item()/len(args.gpu)
            avgkspace_test_l2 = (sum([x[5] for x in collected_test_losses]).item()/len(args.gpu))**0.5
            avgispace_test_real_loss = sum([x[6] for x in collected_test_losses]).item()/len(args.gpu)
            avgispace_test_ssim = sum([x[7] for x in collected_test_losses]).item()/len(args.gpu)
            avgispace_test_l1 = sum([x[8] for x in collected_test_losses]).item()/len(args.gpu)
            avgispace_test_l2 = (sum([x[9] for x in collected_test_losses]).item()/len(args.gpu))**0.5
            if args.neptune_log and rank == 0:
                run["test/kspacetest_mag_loss"] = avgkspace_test_mag_loss
                run["test/kspacetest_phase_loss"] = avgkspace_test_phase_loss
                run["test/kspacetest_real_loss"] = avgkspace_test_real_loss
                run["test/kspacetest_ssim_score"] = avgkspace_test_ssim
                run["test/kspacetest_l1_loss"] = avgkspace_test_l1
                run["test/kspacetest_l2_loss"] = avgkspace_test_l2
                run["test/ispacetest_real_loss"] = avgispace_test_real_loss
                run["test/ispacetest_ssim_score"] = avgispace_test_ssim
                run["test/ispacetest_l1_loss"] = avgispace_test_l1
                run["test/ispacetest_l2_loss"] = avgispace_test_l2

            print('KSpace Test Losses After Epoch {}:'.format(pre_e), flush = True)
            print('KSpace Mag Loss = {}' .format(avgkspace_test_mag_loss), flush = True)
            print('KSpace Phase Loss = {}' .format(avgkspace_test_phase_loss), flush = True)
            print('KSpace Real Loss = {}' .format(avgkspace_test_real_loss), flush = True)
            print('KSpace SSIM = {}' .format(avgkspace_test_ssim), flush = True)
            print('ISpace Test Losses:', flush = True)
            print('ISpace Real (L1) Loss = {}' .format(avgispace_test_real_loss), flush = True)
            print('ISpace SSIM = {}\n\n' .format(avgispace_test_ssim), flush = True)

        if not args.test_only:
            train_kspaceloss_mag, train_kspaceloss_phase, train_kspaceloss_real, train_kspacessim, train_kspaceloss_l1, train_kspaceloss_l2, train_ispaceloss_real, train_ispacessim, train_ispaceloss_l1, train_ispaceloss_l2 = trainer.evaluate(pre_e, train = True)
            collected_train_losses = [torch.zeros(10,).to(proc_device) for _ in range(world_size)]
            dist.all_gather(collected_train_losses, torch.tensor([train_kspaceloss_mag, train_kspaceloss_phase, train_kspaceloss_real, train_kspacessim, train_kspaceloss_l1, train_kspaceloss_l2, train_ispaceloss_real, train_ispacessim, train_ispaceloss_l1, train_ispaceloss_l2]).to(proc_device))
            if rank == 0:
                avgkspace_train_mag_loss = sum([x[0] for x in collected_train_losses]).item()/len(args.gpu)
                avgkspace_train_phase_loss = sum([x[1] for x in collected_train_losses]).item()/len(args.gpu)
                avgkspace_train_real_loss = sum([x[2] for x in collected_train_losses]).item()/len(args.gpu)
                avgkspace_train_ssim = sum([x[3] for x in collected_train_losses]).item()/len(args.gpu)
                avgkspace_train_l1 = sum([x[4] for x in collected_train_losses]).item()/len(args.gpu)
                avgkspace_train_l2 = (sum([x[5] for x in collected_train_losses]).item()/len(args.gpu))**0.5
                avgispace_train_real_loss = sum([x[6] for x in collected_train_losses]).item()/len(args.gpu)
                avgispace_train_ssim = sum([x[7] for x in collected_train_losses]).item()/len(args.gpu)
                avgispace_train_l1 = sum([x[8] for x in collected_train_losses]).item()/len(args.gpu)
                avgispace_train_l2 = (sum([x[9] for x in collected_train_losses]).item()/len(args.gpu))**0.5
                if args.neptune_log and rank == 0:
                    run["test/kspacetrain_mag_loss"] = avgkspace_train_mag_loss
                    run["test/kspacetrain_phase_loss"] = avgkspace_train_phase_loss
                    run["test/kspacetrain_real_loss"] = avgkspace_train_real_loss
                    run["test/kspacetrain_ssim_score"] = avgkspace_train_ssim
                    run["test/kspacetrain_l1_loss"] = avgkspace_train_l1
                    run["test/kspacetrain_l2_loss"] = avgkspace_train_l2
                    run["test/ispacetrain_real_loss"] = avgispace_train_real_loss
                    run["test/ispacetrain_ssim_score"] = avgispace_train_ssim
                    run["test/ispacetrain_l1_loss"] = avgispace_train_l1
                    run["test/ispacetrain_l2_loss"] = avgispace_train_l2

                print('KSpace Test Losses After Epoch {}:'.format(pre_e), flush = True)
                print('KSpace Mag Loss = {}' .format(avgkspace_train_mag_loss), flush = True)
                print('KSpace Phase Loss = {}' .format(avgkspace_train_phase_loss), flush = True)
                print('KSpace Real Loss = {}' .format(avgkspace_train_real_loss), flush = True)
                print('KSpace SSIM = {}' .format(avgkspace_train_ssim), flush = True)
                print('ISpace Test Losses:', flush = True)
                print('ISpace Real (L1) Loss = {}' .format(avgispace_train_real_loss), flush = True)
                print('ISpace SSIM = {}\n\n' .format(avgispace_train_ssim), flush = True)

    if rank == 0:
        with open(os.path.join(args.run_id, 'status.txt'), 'w') as f:
            f.write('1')

    cleanup()