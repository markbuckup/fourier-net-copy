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
    """
    Sets up the process group for distributed training.

    Parameters:
    rank (int): The rank of the current process.
    world_size (int): The total number of processes.
    args (Namespace): Arguments containing the port number and GPU configuration.

    This function initializes the process group using either 'gloo' or 'nccl' backend
    based on the GPU configuration provided in args. It also sets the master address
    and port for communication.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '{}'.format(args.port)
    if args.gpu[0] == -1:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """
    Cleans up the process group for distributed training.

    This function destroys the process group, ensuring that all processes are properly
    terminated and resources are released.
    """
    dist.destroy_process_group()

sys.path.append('../')

SAVE_INTERVAL = 1

def train_paradigm(rank, world_size, args, parameters):
    """
    Trains a distributed model for cardiac MRI reconstruction using DDP.

    Parameters:
    rank (int): The rank of the current process.
    world_size (int): The total number of processes.
    args (Namespace): Arguments containing paths, GPU configuration, logging options, etc.
    parameters (dict): Training parameters including dataset configuration, model parameters, and training options.

    This function sets up the environment for distributed training, initializes the dataset and models,
    loads checkpoint if resuming, and trains the model over a specified number of epochs. Training and
    validation losses are logged, and model checkpoints are saved periodically.
    """
    torch.cuda.set_device(args.gpu[rank])
    setup(rank, world_size, args)
    proc_device = torch.device('cuda:{}'.format(args.gpu[rank]))
    
    if parameters['dataset'] == 'acdc':
        from utils.myDatasets.ACDC_radial_faster import ACDC_radial as dataset
        from utils.myDatasets.ACDC_radial_faster import ACDC_radial_ispace as dataset_ispace
    from utils.models.periodLSTM import fetch_models as rnn_func
    Model_Kspace, Model_Ispace = rnn_func(parameters)
    from utils.Trainers.DDP_LSTMTrainer_nufft import Trainer

    temp = os.getcwd().split('/')
    temp = temp[temp.index('experiments'):]
    save_path = os.path.join(parameters['save_folder'], '/'.join(temp))
    save_path = os.path.join(save_path, args.run_id)

    trainset = dataset(
                        args.dataset_path, 
                        parameters.copy(),
                        proc_device,
                        train = True, 
                    )
    testset = dataset(
                        args.dataset_path, 
                        parameters.copy(),
                        proc_device,
                        train = False, 
                    )
    if parameters['num_epochs_unet'] == 0 or (not parameters['memoise_ispace']):
        ispace_trainset = None
        ispace_testset = None
    else:
        ispace_trainset = dataset_ispace(
                            args.dataset_path, 
                            parameters.copy(),
                            proc_device,
                            train = True, 
                        )
        ispace_testset = dataset_ispace(
                            args.dataset_path, 
                            parameters.copy(),
                            proc_device,
                            train = False, 
                        )

    recurrent_model = Model_Kspace(parameters, proc_device).to(proc_device)
    coil_combine_unet = Model_Ispace(parameters, proc_device).to(proc_device)
    
    checkpoint_path = os.path.join(save_path, 'checkpoints/')
    os.makedirs(checkpoint_path, exist_ok = True)

    parameters['GPUs'] = args.gpu
    if rank == 0:
        if args.neptune_log:
            if os.path.isfile(checkpoint_path + 'neptune_run.pth'):
                run_id = torch.load(checkpoint_path + 'neptune_run.pth', map_location = torch.device('cpu'))['run_id']
                run = neptune.init_run(
                    project="fcrl/Cardiac-MRI-Reconstruction",
                    custom_run_id=run_id,
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
            if not (args.resume or args.resume_kspace):
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
        recurrent_model.load_state_dict(dic['recurrent_model'])
        coil_combine_unet.load_state_dict(dic['coil_combine_unet'])
        opt_dict_recurrent = dic['recurrent_optim']
        opt_dict_unet = dic['unet_optim']
        if parameters['scheduler'] != 'None':
            scheduler_dict_unet = dic['unet_scheduler']
            scheduler_dict_recurrent = dic['recurrent_scheduler']
        losses = dic['losses']
        test_losses = dic['test_losses']
        if rank == 0:
            print('Resuming Training after {} epochs'.format(pre_e), flush = True)
        del dic
    elif args.resume_kspace:
        model_state = 0
        if rank == 0:
            print('Loading checkpoint at model state {}'.format(model_state), flush = True)
        dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state),map_location = proc_device)
        pre_e = dic['e']
        recurrent_model.load_state_dict(dic['recurrent_model'])
        # coil_combine_unet.load_state_dict(dic['coil_combine_unet'])
        opt_dict_recurrent = dic['recurrent_optim']
        # opt_dict_unet = dic['unet_optim']
        if parameters['scheduler'] != 'None':
            scheduler_dict_unet = dic['unet_scheduler']
            scheduler_dict_recurrent = dic['recurrent_scheduler']
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
        opt_dict_recurrent = None
        opt_dict_unet = None
        scheduler_dict_unet = None
        scheduler_dict_recurrent = None
        if rank == 0:
            print('Starting Training', flush = True)

    recurrent_model = DDP(recurrent_model, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = False)
    coil_combine_unet = DDP(coil_combine_unet, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = False)
    trainer = Trainer(recurrent_model, coil_combine_unet, ispace_trainset, ispace_testset, trainset, testset, parameters, proc_device, rank, world_size, args)

    if args.time_analysis:
        if rank == 0:
            trainer.time_analysis()
        return

    if args.resume or args.resume_kspace:
        if not args.resume_kspace:
            trainer.unet_optim.load_state_dict(opt_dict_unet)
        trainer.recurrent_optim.load_state_dict(opt_dict_recurrent)
        if parameters['scheduler'] != 'None':
            if not args.resume_kspace:
                trainer.unet_scheduler.load_state_dict(scheduler_dict_unet)
            trainer.recurrent_scheduler.load_state_dict(scheduler_dict_recurrent)

    for e in range(parameters['num_epochs_total']):
        if pre_e > 0:
            pre_e -= 1
            continue
        collected_train_losses = [torch.zeros(15,).to(proc_device) for _ in range(world_size)]
        collected_test_losses = [torch.zeros(15,).to(proc_device) for _ in range(world_size)]
        
        kspaceloss_mag, kpsaceloss_phase, kspaceloss_real, loss_forget_gate, loss_input_gate, kspacessim, kpsaceloss_l1, kspaceloss_l2, sosssim_score, sosl1_score, sosl2_score, ispaceloss_real, ispacessim, ipsaceloss_l1, ispaceloss_l2 = trainer.train(e)
        dist.all_gather(collected_train_losses, torch.tensor([kspaceloss_mag, kpsaceloss_phase, kspaceloss_real, loss_forget_gate, loss_input_gate, kspacessim, kpsaceloss_l1, kspaceloss_l2, sosssim_score, sosl1_score, sosl2_score, ispaceloss_real, ispacessim, ipsaceloss_l1, ispaceloss_l2]).to(proc_device))
        if rank == 0:
            avgkspace_train_mag_loss = sum([x[0] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            avgkspace_train_phase_loss = sum([x[1] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            avgkspace_train_real_loss = sum([x[2] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            avgkspace_train_forget_gate_loss = sum([x[3] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            avgkspace_train_input_gate_loss = sum([x[4] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            kspacessim_score = sum([x[5] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            kspacel1_score = sum([x[6] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            kspacel2_score = (sum([x[7] for x in collected_train_losses]).cpu().item()/len(args.gpu))**0.5
            sosssim_score = sum([x[8] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            sosl1_score = sum([x[9] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            sosl2_score = (sum([x[10] for x in collected_train_losses]).cpu().item()/len(args.gpu))**0.5
            avgispace_train_real_loss = sum([x[11] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            ispacessim_score = sum([x[12] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            ispacel1_score = sum([x[13] for x in collected_train_losses]).cpu().item()/len(args.gpu)
            ispacel2_score = (sum([x[14] for x in collected_train_losses]).cpu().item()/len(args.gpu))**0.5
            if args.neptune_log and rank == 0:
                run["train/epochs_trained"].log(e)
                run["train/kspace_train_mag_loss"].log(avgkspace_train_mag_loss)
                run["train/kspace_train_phase_loss"].log(avgkspace_train_phase_loss)
                run["train/kspace_train_real_loss"].log(avgkspace_train_real_loss)
                run["train/kspace_train_forget_gate_loss"].log(avgkspace_train_forget_gate_loss)
                run["train/kspace_train_input_gate_loss"].log(avgkspace_train_input_gate_loss)
                run["train/kspace_train_ssim"].log(kspacessim_score)
                run["train/kspace_train_l1"].log(kspacel1_score)
                run["train/kspace_train_l2"].log(kspacel2_score)
                run["train/sos_train_ssim"].log(sosssim_score)
                run["train/sos_train_l1"].log(sosl1_score)
                run["train/sos_train_l2"].log(sosl2_score)
                run["train/ispace_train_real_loss"].log(avgispace_train_real_loss)
                run["train/ispace_train_ssim"].log(ispacessim_score)
                run["train/ispace_train_l1"].log(ispacel1_score)
                run["train/ispace_train_l2"].log(ispacel2_score)
            
            print('KSpace Training Losses for Epoch {}:'.format(e), flush = True)
            print('KSpace Mag Loss = {}' .format(avgkspace_train_mag_loss), flush = True)
            print('KSpace Phase Loss = {}' .format(avgkspace_train_phase_loss), flush = True)
            print('KSpace Real Loss = {}' .format(avgkspace_train_real_loss), flush = True)
            print('KSpace Forget Gate Loss = {}' .format(avgkspace_train_forget_gate_loss), flush = True)
            print('KSpace Input Gate Loss = {}' .format(avgkspace_train_input_gate_loss), flush = True)
            print('KSpace SSIM = {}' .format(kspacessim_score), flush = True)
            print('\nSOS Training Losses:', flush = True)
            print('SOS L1 = {}' .format(sosl1_score), flush = True)
            print('SOS L2 = {}' .format(sosl2_score), flush = True)
            print('SOS SSIM = {}' .format(sosssim_score), flush = True)
            print('\nISpace Training Losses:', flush = True)
            print('ISpace Real (L1) Loss = {}' .format(avgispace_train_real_loss), flush = True)
            print('ISpace SSIM = {}\n\n' .format(ispacessim_score), flush = True)

        kspaceloss_mag, kpsaceloss_phase, kspaceloss_real, loss_forget_gate, loss_input_gate, kspacessim, kpsaceloss_l1, kspaceloss_l2, test_sosssim_score, test_sosl1_score, test_sosl2_score, ispaceloss_real, ispacessim, ipsaceloss_l1, ispaceloss_l2 = trainer.evaluate(e, train = False)
        dist.all_gather(collected_test_losses, torch.tensor([kspaceloss_mag, kpsaceloss_phase, kspaceloss_real, loss_forget_gate, loss_input_gate, kspacessim, kpsaceloss_l1, kspaceloss_l2, test_sosssim_score, test_sosl1_score, test_sosl2_score, ispaceloss_real, ispacessim, ipsaceloss_l1, ispaceloss_l2]).to(proc_device))
        if rank == 0:
            avgkspace_test_mag_loss = sum([x[0] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            avgkspace_test_phase_loss = sum([x[1] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            avgkspace_test_real_loss = sum([x[2] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            avgkspace_test_forget_gate_loss = sum([x[3] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            avgkspace_test_input_gate_loss = sum([x[4] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_kspacessim_score = sum([x[5] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_kspacel1_score = sum([x[6] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_kspacel2_score = (sum([x[7] for x in collected_test_losses]).cpu().item()/len(args.gpu)) ** 0.5
            test_sosssim_score = sum([x[8] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_sosl1_score = sum([x[9] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_sosl2_score = (sum([x[10] for x in collected_test_losses]).cpu().item()/len(args.gpu)) ** 0.5
            avgispace_test_real_loss = sum([x[11] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_ispacessim_score = sum([x[12] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_ispacel1_score = sum([x[13] for x in collected_test_losses]).cpu().item()/len(args.gpu)
            test_ispacel2_score = (sum([x[14] for x in collected_test_losses]).cpu().item()/len(args.gpu)) ** 0.5
            if args.neptune_log and rank == 0:
                run["train/kspacetest_mag_loss"].log(avgkspace_test_mag_loss)
                run["train/kspacetest_phase_loss"].log(avgkspace_test_phase_loss)
                run["train/kspacetest_real_loss"].log(avgkspace_test_real_loss)
                run["train/kspacetest_forget_gate_loss"].log(avgkspace_test_forget_gate_loss)
                run["train/kspacetest_input_gate_loss"].log(avgkspace_test_input_gate_loss)
                run["train/kspacetest_ssim"].log(test_kspacessim_score)
                run["train/kspacetest_l1"].log(test_kspacel1_score)
                run["train/kspacetest_l2"].log(test_kspacel2_score)
                run["train/sostest_ssim"].log(test_sosssim_score)
                run["train/sostest_l1"].log(test_sosl1_score)
                run["train/sostest_l2"].log(test_sosl2_score)
                run["train/ispacetest_real_loss"].log(avgispace_test_real_loss)
                run["train/ispacetest_ssim"].log(test_ispacessim_score)
                run["train/ispacetest_l1"].log(test_ispacel1_score)
                run["train/ispacetest_l2"].log(test_ispacel2_score)
            
            print('KSpace Test Losses for Epoch {}:'.format(e), flush = True)
            print('KSpace Mag Loss = {}' .format(avgkspace_test_mag_loss), flush = True)
            print('KSpace Phase Loss = {}' .format(avgkspace_test_phase_loss), flush = True)
            print('KSpace Real Loss = {}' .format(avgkspace_test_real_loss), flush = True)
            print('KSpace Forget Gate Loss = {}' .format(avgkspace_test_forget_gate_loss), flush = True)
            print('KSpace Input Gate Loss = {}' .format(avgkspace_test_input_gate_loss), flush = True)
            print('KSpace SSIM = {}' .format(test_kspacessim_score), flush = True)
            print('\nSOS Test Losses:', flush = True)
            print('SOS L1 = {}' .format(test_sosl1_score), flush = True)
            print('SOS L2 = {}' .format(test_sosl2_score), flush = True)
            print('SOS SSIM = {}' .format(test_sosssim_score), flush = True)
            print('\nISpace Test Losses:', flush = True)
            print('ISpace Real (L1) Loss = {}' .format(avgispace_test_real_loss), flush = True)
            print('ISpace SSIM = {}' .format(test_ispacessim_score), flush = True)
            
        if rank == 0:
            losses.append((avgkspace_train_mag_loss,avgkspace_train_phase_loss,avgkspace_train_real_loss,avgkspace_train_forget_gate_loss,avgkspace_train_input_gate_loss,kspacessim_score,kspacel1_score,kspacel2_score,sosssim_score,sosl1_score,sosl2_score,avgispace_train_real_loss,ispacessim_score,ispacel1_score,ispacel2_score))
            test_losses.append((avgkspace_test_mag_loss, avgkspace_test_phase_loss, avgkspace_test_real_loss, avgkspace_test_forget_gate_loss, avgkspace_test_input_gate_loss, test_kspacessim_score, test_kspacel1_score, test_kspacel2_score, test_sosssim_score, test_sosl1_score, test_sosl2_score, avgispace_test_real_loss, test_ispacessim_score, test_ispacel1_score, test_ispacel2_score))

            parameters['train_losses'] = losses
            parameters['test_losses'] = test_losses

            dic = {}
            dic['e'] = e+1
            dic['recurrent_model'] = trainer.recurrent_model.module.state_dict()
            dic['coil_combine_unet'] = trainer.coil_combine_unet.module.state_dict()
            dic['recurrent_optim'] = trainer.recurrent_optim.state_dict()

            dic['unet_optim'] = trainer.unet_optim.state_dict()
            if parameters['scheduler'] != 'None':
                dic['unet_scheduler'] = trainer.unet_scheduler.state_dict()
                dic['recurrent_scheduler'] = trainer.recurrent_scheduler.state_dict()
            dic['losses'] = losses
            dic['test_losses'] = test_losses
            if (e+1) % SAVE_INTERVAL == 0:
                if e > parameters['num_epochs_recurrent']:
                    model_state = 3
                elif e > parameters['num_epochs_recurrent'] - parameters['num_epochs_windowed']:
                    model_state = 2
                elif e > parameters['num_epochs_recurrent'] - parameters['num_epochs_ilstm']:
                    model_state = 1
                else:
                    model_state = 0
                torch.save(dic, checkpoint_path + 'checkpoint_{}.pth'.format(model_state))
                torch.save({'state': model_state}, checkpoint_path + 'state.pth')
                # model_state += 1
                print('Saving model after {} Epochs\n\n'.format(e+1), flush = True)
                print('##########################################################################################')
            del dic
        del collected_test_losses
        del collected_train_losses
        torch.cuda.empty_cache()

    if rank == 0:
        with open(os.path.join(args.run_id, 'status.txt'), 'w') as f:
            f.write('1')

    cleanup()

"""
    Tests a distributed model for cardiac MRI reconstruction using DDP.

    Parameters:
    rank (int): The rank of the current process.
    world_size (int): The total number of processes.
    args (Namespace): Arguments containing paths, GPU configuration, logging options, etc.
    parameters (dict): Testing parameters including dataset configuration, model parameters, and testing options.

    This function sets up the environment for distributed testing, initializes the dataset and models,
    loads checkpoint if resuming, and evaluates the model on the test dataset. Test losses are logged,
    and model performance is visualized if specified.
    """
def test_paradigm(rank, world_size, args, parameters):
    torch.cuda.set_device(args.gpu[rank])
    setup(rank, world_size, args)
    proc_device = torch.device('cuda:{}'.format(args.gpu[rank]))
    if parameters['dataset'] == 'acdc':
        from utils.myDatasets.ACDC_radial_faster import ACDC_radial as dataset
    from utils.models.periodLSTM import fetch_models as rnn_func
    Model_Kspace, Model_Ispace = rnn_func(parameters)
    from utils.Trainers.DDP_LSTMTrainer_nufft import Trainer

    temp = os.getcwd().split('/')
    temp = temp[temp.index('experiments'):]
    save_path = os.path.join(parameters['save_folder'], '/'.join(temp))
    save_path = os.path.join(save_path, args.run_id)

    if not args.eval_on_real:
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
    else:
        trainset = None
        testset = None

    ispace_trainset = None
    ispace_testset = None



    recurrent_model = Model_Kspace(parameters, proc_device).to(proc_device)
    coil_combine_unet = Model_Ispace(parameters, proc_device).to(proc_device)
    checkpoint_path = os.path.join(save_path, 'checkpoints/')

    if rank == 0:
        if args.neptune_log:
            if os.path.isfile(checkpoint_path + 'neptune_run.pth'):
                run_id = torch.load(checkpoint_path + 'neptune_run.pth', map_location = torch.device('cpu'))['run_id']
                run = neptune.init_run(
                    project="fcrl/Cardiac-MRI-Reconstruction",
                    custom_run_id=run_id,
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
            if not args.resume or args.resume_kspace:
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
        recurrent_model.load_state_dict(dic['recurrent_model'])
        coil_combine_unet.load_state_dict(dic['coil_combine_unet'])
        if rank == 0:
            print('Loading kspace model after {} epochs'.format(dic['e']), flush = True)
        losses = dic['losses']
        test_losses = dic['test_losses']
        if rank == 0:
            print('Resuming Training after {} epochs'.format(pre_e), flush = True)
        del dic
    elif args.resume_kspace:
        model_state = 0
        if (not args.state == -1):
            model_state = args.state
        if rank == 0:
            print('Loading checkpoint at model state {}'.format(model_state), flush = True)
        dic = torch.load(checkpoint_path + 'checkpoint_{}.pth'.format(model_state),map_location = proc_device)
        pre_e = dic['e']
        recurrent_model.load_state_dict(dic['recurrent_model'])
        # coil_combine_unet.load_state_dict(dic['coil_combine_unet'])
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
        scheduler_dict = None
        if rank == 0:
            print('Starting Training', flush = True)

    # recurrent_model = DDP(recurrent_model, device_ids = None, output_device = None, find_unused_parameters = False)
    recurrent_model = DDP(recurrent_model, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = False)
    coil_combine_unet = DDP(coil_combine_unet, device_ids = [args.gpu[rank]], output_device = args.gpu[rank], find_unused_parameters = False)
    trainer = Trainer(recurrent_model, coil_combine_unet, ispace_trainset, ispace_testset, trainset, testset, parameters, proc_device, rank, world_size, args)

    if args.time_analysis:
        if rank == 0:
            trainer.time_analysis()
        return
    
    if args.eval_on_real:
        if rank == 0:
            if not args.numbers_only:
                trainer.visualise_on_real(pre_e)
        cleanup()
        return

    if rank == 0:
        # if args.neptune_log and rank == 0:
        #     for x in sorted(os.listdir(os.path.join(args.run_id, 'images/train'))):
        #         run['train/{}'.format(x)].upload(File(os.path.join(args.run_id, 'images/train/{}'.format(x))))
        #         break
        #     for x in sorted(os.listdir(os.path.join(args.run_id, 'images/test'))):
        #         run['test/{}'.format(x)].upload(File(os.path.join(args.run_id, 'images/test/{}'.format(x))))
        #         break

        # train_losses contain
                    # train_kspace_mag_loss
                    # train_kspace_phase_loss
                    # train_kspace_real_loss
                    # train_kspace_forget_gate_loss
                    # train_kspace_input_gate_loss
                    # train_kspace_ssim_score
                    # train_kspace_l1_score
                    # train_kspace_l2_score
                    # train_sos_ssim_score
                    # train_sos_l1_score
                    # train_sos_l2_score
                    # train_ispace_real_loss
                    # train_ispace_ssim_score
                    # train_ispace_l1_score
                    # train_ispace_l2_score
        # test_losses contain 
                    # test_kspace_mag_loss
                    # test_kspace_phase_loss
                    # test_kspace_real_loss
                    # test_kspace_forget_gate_loss
                    # test_kspace_input_gate_loss
                    # test_kspace_ssim_score
                    # test_kspace_l1_score
                    # test_kspace_l2_score
                    # test_sos_ssim_score
                    # test_sos_l1_score
                    # test_sos_l2_score
                    # test_ispace_real_loss
                    # test_ispace_ssim_score
                    # test_ispace_l1_score
                    # test_ispace_l2_score
        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)

        kspace_x = range(min(parameters['num_epochs_recurrent'], len(losses)))
        ispace_x = range(max(parameters['num_epochs_total'] - parameters['num_epochs_unet']+1, 0), min(parameters['num_epochs_total'], len(losses)))
        
        plt.figure()
        plt.title('Train Kspace Mag Loss after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[0] for i,x in enumerate(losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_kspace_mag_loss.png'))
        plt.figure()
        plt.title('Train Kspace Phase Loss after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[1] for i,x in enumerate(losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_kspace_phase_loss.png'))
        plt.figure()
        plt.title('Train Kspace Real Loss after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[2] for i,x in enumerate(losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_kspace_real_loss.png'))
        plt.figure()
        plt.title('Train Kspace Forget gate Loss after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[3] for i,x in enumerate(losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_kspace_forget_gate_loss.png'))
        plt.figure()
        plt.title('Train Kspace Input gate Loss after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[4] for i,x in enumerate(losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_kspace_input_gate_loss.png'))
        plt.figure()
        plt.title('Train Kspace SSIM score after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[5] for i,x in enumerate(losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_kspace_ssim_score.png'))
        plt.figure()
        plt.title('Train Kspace L1 score after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[6] for i,x in enumerate(losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_kspace_l1_score.png'))
        plt.figure()
        plt.title('Train Kspace L2 score after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[7] for i,x in enumerate(losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_kspace_l2_score.png'))
        plt.figure()
        plt.title('Train SOS ssim score after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[8] for i,x in enumerate(losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_sos_ssim_score.png'))
        plt.figure()
        plt.title('Train SOS L1 score after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[9] for i,x in enumerate(losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_sos_l1_score.png'))
        plt.figure()
        plt.title('Train SOS L2 score after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[10] for i,x in enumerate(losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_sos_l2_score.png'))
        plt.figure()
        plt.title('Train Ispace Real Loss after {} epochs'.format(pre_e))
        plt.plot(ispace_x, [x[11] for i,x in enumerate(losses) if i > (parameters['num_epochs_total'] - parameters['num_epochs_unet'])], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_ispace_real_loss.png'))
        plt.figure()
        plt.title('Train Ispace SSIM score after {} epochs'.format(pre_e))
        plt.plot(ispace_x, [x[12] for i,x in enumerate(losses) if i > (parameters['num_epochs_total'] - parameters['num_epochs_unet'])], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_ispace_ssim_score.png'))
        plt.figure()
        plt.title('Train Ispace L1 score after {} epochs'.format(pre_e))
        plt.plot(ispace_x, [x[13] for i,x in enumerate(losses) if i > (parameters['num_epochs_total'] - parameters['num_epochs_unet'])], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_ispace_l1_score.png'))
        plt.figure()
        plt.title('Train Ispace L2 score after {} epochs'.format(pre_e))
        plt.plot(ispace_x, [x[14] for i,x in enumerate(losses) if i > (parameters['num_epochs_total'] - parameters['num_epochs_unet'])], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/train_ispace_l2_score.png'))
        
        plt.close('all')

        plt.figure()
        plt.title('Test Kspace Mag Loss after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[0] for i,x in enumerate(test_losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_kspace_mag_loss.png'))
        plt.figure()
        plt.title('Test Kspace Phase Loss after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[1] for i,x in enumerate(test_losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_kspace_phase_loss.png'))
        plt.figure()
        plt.title('Test Kspace Real Loss after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[2] for i,x in enumerate(test_losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_kspace_real_loss.png'))
        plt.figure()
        plt.title('Test Kspace Forget gate Loss after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[3] for i,x in enumerate(test_losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_kspace_forget_gate_loss.png'))
        plt.figure()
        plt.title('Test Kspace Input gate Loss after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[4] for i,x in enumerate(test_losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_kspace_input_gate_loss.png'))
        plt.figure()
        plt.title('Test Kspace SSIM score after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[5] for i,x in enumerate(test_losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_kspace_ssim_score.png'))
        plt.figure()
        plt.title('Test Kspace L1 score after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[6] for i,x in enumerate(test_losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_kspace_l1_score.png'))
        plt.figure()
        plt.title('Test Kspace L2 score after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[7] for i,x in enumerate(test_losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_kspace_l2_score.png'))
        plt.figure()
        plt.title('Test SOS ssim score after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[8] for i,x in enumerate(test_losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_sos_ssim_score.png'))
        plt.figure()
        plt.title('Test SOS L1 score after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[9] for i,x in enumerate(test_losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_sos_l1_score.png'))
        plt.figure()
        plt.title('Test SOS L2 score after {} epochs'.format(pre_e))
        plt.plot(kspace_x, [x[10] for i,x in enumerate(test_losses) if i < parameters['num_epochs_recurrent']], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_sos_l2_score.png'))
        plt.figure()
        plt.title('Test Ispace Real Loss after {} epochs'.format(pre_e))
        plt.plot(ispace_x, [x[11] for i,x in enumerate(test_losses) if i > (parameters['num_epochs_total'] - parameters['num_epochs_unet'])], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_ispace_real_loss.png'))
        plt.figure()
        plt.title('Test Ispace SSIM score after {} epochs'.format(pre_e))
        plt.plot(ispace_x, [x[12] for i,x in enumerate(test_losses) if i > (parameters['num_epochs_total'] - parameters['num_epochs_unet'])], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_ispace_ssim_score.png'))
        plt.figure()
        plt.title('Test Ispace L1 score after {} epochs'.format(pre_e))
        plt.plot(ispace_x, [x[13] for i,x in enumerate(test_losses) if i > (parameters['num_epochs_total'] - parameters['num_epochs_unet'])], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_ispace_l1_score.png'))
        plt.figure()
        plt.title('Test Ispace L2 score after {} epochs'.format(pre_e))
        plt.plot(ispace_x, [x[14] for i,x in enumerate(test_losses) if i > (parameters['num_epochs_total'] - parameters['num_epochs_unet'])], color = 'b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(save_path, 'images/test_ispace_l2_score.png'))
        

        plt.close('all')
        if not args.numbers_only:
            trainer.visualise(pre_e, train = False)
            trainer.visualise(pre_e, train = True)

    if not args.visualise_only:
        test_kspaceloss_mag, test_kspaceloss_phase, test_kspaceloss_real, test_loss_forget_gate, test_loss_input_gate, test_kspacessim, test_kspaceloss_l1, test_kspaceloss_l2, avgsos_test_ssim, avgsos_test_l1, avgsos_test_l2, test_ispaceloss_real, test_ispacessim, test_ispaceloss_l1, test_ispaceloss_l2 = trainer.evaluate(pre_e, train = False)
        collected_test_losses = [torch.zeros(15,).to(proc_device) for _ in range(world_size)]
        dist.all_gather(collected_test_losses, torch.tensor([test_kspaceloss_mag, test_kspaceloss_phase, test_kspaceloss_real, test_loss_forget_gate, test_loss_input_gate, test_kspacessim, test_kspaceloss_l1, test_kspaceloss_l2, avgsos_test_ssim, avgsos_test_l1, avgsos_test_l2, test_ispaceloss_real, test_ispacessim, test_ispaceloss_l1, test_ispaceloss_l2]).to(proc_device))
        if rank == 0:
            avgkspace_test_mag_loss = sum([x[0] for x in collected_test_losses]).item()/len(args.gpu)
            avgkspace_test_phase_loss = sum([x[1] for x in collected_test_losses]).item()/len(args.gpu)
            avgkspace_test_real_loss = sum([x[2] for x in collected_test_losses]).item()/len(args.gpu)
            avgkspace_test_forget_gate_loss = sum([x[3] for x in collected_test_losses]).item()/len(args.gpu)
            avgkspace_test_input_gate_loss = sum([x[4] for x in collected_test_losses]).item()/len(args.gpu)
            avgkspace_test_ssim = sum([x[5] for x in collected_test_losses]).item()/len(args.gpu)
            avgkspace_test_l1 = sum([x[6] for x in collected_test_losses]).item()/len(args.gpu)
            avgkspace_test_l2 = (sum([x[7] for x in collected_test_losses]).item()/len(args.gpu))**0.5
            avgsos_test_ssim = sum([x[8] for x in collected_test_losses]).item()/len(args.gpu)
            avgsos_test_l1 = sum([x[9] for x in collected_test_losses]).item()/len(args.gpu)
            avgsos_test_l2 = (sum([x[10] for x in collected_test_losses]).item()/len(args.gpu))**0.5
            avgispace_test_real_loss = sum([x[11] for x in collected_test_losses]).item()/len(args.gpu)
            avgispace_test_ssim = sum([x[12] for x in collected_test_losses]).item()/len(args.gpu)
            avgispace_test_l1 = sum([x[12] for x in collected_test_losses]).item()/len(args.gpu)
            avgispace_test_l2 = (sum([x[14] for x in collected_test_losses]).item()/len(args.gpu))**0.5
            if args.neptune_log and rank == 0:
                run["test/kspacetest_mag_loss"].log(avgkspace_test_mag_loss)
                run["test/kspacetest_phase_loss"].log(avgkspace_test_phase_loss)
                run["test/kspacetest_real_loss"].log(avgkspace_test_real_loss)
                run["test/kspacetest_forget_gate_loss"].log(avgkspace_test_forget_gate_loss)
                run["test/kspacetest_input_gate_loss"].log(avgkspace_test_input_gate_loss)
                run["test/kspacetest_ssim_score"].log(avgkspace_test_ssim)
                run["test/kspacetest_l1_loss"].log(avgkspace_test_l1)
                run["test/kspacetest_l2_loss"].log(avgkspace_test_l2)
                run["test/sostest_ssim_score"].log(avgsos_test_ssim)
                run["test/sostest_l1_loss"].log(avgsos_test_l1)
                run["test/sostest_l2_loss"].log(avgsos_test_l2)
                run["test/ispacetest_real_loss"].log(avgispace_test_real_loss)
                run["test/ispacetest_ssim_score"].log(avgispace_test_ssim)
                run["test/ispacetest_l1_loss"].log(avgispace_test_l1)
                run["test/ispacetest_l2_loss"].log(avgispace_test_l2)


            print('KSpace Test Losses After Epoch {}:'.format(pre_e), flush = True)
            print('KSpace Mag Loss = {}' .format(avgkspace_test_mag_loss), flush = True)
            print('KSpace Phase Loss = {}' .format(avgkspace_test_phase_loss), flush = True)
            print('KSpace Real Loss = {}' .format(avgkspace_test_real_loss), flush = True)
            print('KSpace Forget Gate Loss = {}' .format(avgkspace_test_forget_gate_loss), flush = True)
            print('KSpace Input Gate Loss = {}' .format(avgkspace_test_input_gate_loss), flush = True)
            print('KSpace L1 = {}' .format(avgkspace_test_l1), flush = True)
            print('KSpace L2 = {}' .format(avgkspace_test_l2), flush = True)
            print('KSpace SSIM = {}' .format(avgkspace_test_ssim), flush = True)
            print('\nSOS Test Losses:', flush = True)
            print('SOS L1 = {}' .format(avgsos_test_l1), flush = True)
            print('SOS L2 = {}' .format(avgsos_test_l2), flush = True)
            print('SOS SSIM = {}' .format(avgsos_test_ssim), flush = True)
            print('\nISpace Test Losses:', flush = True)
            print('ISpace Real (L1) Loss = {}' .format(avgispace_test_real_loss), flush = True)
            print('ISpace L1 = {}' .format(avgispace_test_l1), flush = True)
            print('ISpace L2 = {}' .format(avgispace_test_l2), flush = True)
            print('ISpace SSIM = {}\n\n' .format(avgispace_test_ssim), flush = True)

        if not args.test_only:
            train_kspaceloss_mag, train_kspaceloss_phase, train_kspaceloss_real, train_loss_forget_gate, train_loss_input_gate, train_kspacessim, train_kspaceloss_l1, train_kspaceloss_l2, avgsos_train_ssim, avgsos_train_l1, avgsos_train_l2, train_ispaceloss_real, train_ispacessim, train_ispaceloss_l1, train_ispaceloss_l2 = trainer.evaluate(pre_e, train = True)
            collected_train_losses = [torch.zeros(15,).to(proc_device) for _ in range(world_size)]
            dist.all_gather(collected_train_losses, torch.tensor([train_kspaceloss_mag, train_kspaceloss_phase, train_kspaceloss_real, train_loss_forget_gate, train_loss_input_gate, train_kspacessim, train_kspaceloss_l1, train_kspaceloss_l2, avgsos_train_ssim, avgsos_train_l1, avgsos_train_l2, train_ispaceloss_real, train_ispacessim, train_ispaceloss_l1, train_ispaceloss_l2]).to(proc_device))
            if rank == 0:
                avgkspace_train_mag_loss = sum([x[0] for x in collected_train_losses]).item()/len(args.gpu)
                avgkspace_train_phase_loss = sum([x[1] for x in collected_train_losses]).item()/len(args.gpu)
                avgkspace_train_real_loss = sum([x[2] for x in collected_train_losses]).item()/len(args.gpu)
                avgkspace_train_forget_gate_loss = sum([x[3] for x in collected_train_losses]).item()/len(args.gpu)
                avgkspace_train_input_gate_loss = sum([x[4] for x in collected_train_losses]).item()/len(args.gpu)
                avgkspace_train_ssim = sum([x[5] for x in collected_train_losses]).item()/len(args.gpu)
                avgkspace_train_l1 = sum([x[6] for x in collected_train_losses]).item()/len(args.gpu)
                avgkspace_train_l2 = (sum([x[7] for x in collected_train_losses]).item()/len(args.gpu))**0.5
                avgsos_train_ssim = sum([x[8] for x in collected_train_losses]).item()/len(args.gpu)
                avgsos_train_l1 = sum([x[9] for x in collected_train_losses]).item()/len(args.gpu)
                avgsos_train_l2 = (sum([x[10] for x in collected_train_losses]).item()/len(args.gpu))**0.5
                avgispace_train_real_loss = sum([x[11] for x in collected_train_losses]).item()/len(args.gpu)
                avgispace_train_ssim = sum([x[12] for x in collected_train_losses]).item()/len(args.gpu)
                avgispace_train_l1 = sum([x[13] for x in collected_train_losses]).item()/len(args.gpu)
                avgispace_train_l2 = (sum([x[14] for x in collected_train_losses]).item()/len(args.gpu))**0.5
                if args.neptune_log and rank == 0:
                    run["test/kspacetrain_mag_loss"].log(avgkspace_train_mag_loss)
                    run["test/kspacetrain_phase_loss"].log(avgkspace_train_phase_loss)
                    run["test/kspacetrain_real_loss"].log(avgkspace_train_real_loss)
                    run["test/kspacetrain_forget_gate_loss"].log(avgkspace_train_forget_gate_loss)
                    run["test/kspacetrain_input_gate_loss"].log(avgkspace_train_input_gate_loss)
                    run["test/kspacetrain_ssim_score"].log(avgkspace_train_ssim)
                    run["test/kspacetrain_l1_loss"].log(avgkspace_train_l1)
                    run["test/kspacetrain_l2_loss"].log(avgkspace_train_l2)
                    run["test/sostrain_ssim_score"].log(avgsos_train_ssim)
                    run["test/sostrain_l1_loss"].log(avgsos_train_l1)
                    run["test/sostrain_l2_loss"].log(avgsos_train_l2)
                    run["test/ispacetrain_real_loss"].log(avgispace_train_real_loss)
                    run["test/ispacetrain_ssim_score"].log(avgispace_train_ssim)
                    run["test/ispacetrain_l1_loss"].log(avgispace_train_l1)
                    run["test/ispacetrain_l2_loss"].log(avgispace_train_l2)

                print('KSpace Train Losses After Epoch {}:'.format(pre_e), flush = True)
                print('KSpace Mag Loss = {}' .format(avgkspace_train_mag_loss), flush = True)
                print('KSpace Phase Loss = {}' .format(avgkspace_train_phase_loss), flush = True)
                print('KSpace Real Loss = {}' .format(avgkspace_train_real_loss), flush = True)
                print('KSpace Forget Gate Loss = {}' .format(avgkspace_train_forget_gate_loss), flush = True)
                print('KSpace Input Gate Loss = {}' .format(avgkspace_train_input_gate_loss), flush = True)
                print('KSpace L1 = {}' .format(avgkspace_train_l1), flush = True)
                print('KSpace L2 = {}' .format(avgkspace_train_l2), flush = True)
                print('KSpace SSIM = {}' .format(avgkspace_train_ssim), flush = True)
                print('\nSOS Train Losses:', flush = True)
                print('SOS L1 = {}' .format(avgsos_train_l1), flush = True)
                print('SOS L2 = {}' .format(avgsos_train_l2), flush = True)
                print('SOS SSIM = {}' .format(avgsos_train_ssim), flush = True)
                print('\nISpace Train Losses:', flush = True)
                print('ISpace Real (L1) Loss = {}' .format(avgispace_train_real_loss), flush = True)
                print('ISpace L1 = {}' .format(avgispace_train_l1), flush = True)
                print('ISpace L2 = {}' .format(avgispace_train_l2), flush = True)
                print('ISpace SSIM = {}\n\n' .format(avgispace_train_ssim), flush = True)

    if rank == 0:
        with open(os.path.join(args.run_id, 'status.txt'), 'w') as f:
            f.write('2')

    cleanup()