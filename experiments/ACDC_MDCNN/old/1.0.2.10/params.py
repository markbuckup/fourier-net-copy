parameters = {}
parameters['image_resolution'] = 64
parameters['lr_kspace'] = 1e-5
parameters['lr_ispace'] = 3e-4
parameters['lr_gate'] = 3e-4
parameters['init_skip_frames'] = 10
parameters['num_epochs'] = 50
parameters['architecture'] = 'mdcnn'
parameters['dataset'] = 'acdc'
parameters['train_test_split'] = 0.8
parameters['normalisation'] = False
parameters['window_size'] = 7
parameters['loop_videos'] = -1
parameters['SHM_looping'] = False
if 'gru' in parameters['architecture']:
    batch_sizes = [-1,[6,14],[6,13],[6,30],[7,34],[7,32]]
    parameters['train_batch_size'] = batch_sizes[int(parameters['architecture'][-1])][parameters['image_space_real']]
    parameters['test_batch_size'] = batch_sizes[int(parameters['architecture'][-1])][parameters['image_space_real']]
else:
    if parameters['image_resolution'] == 256:
        parameters['train_batch_size'] = 8
        parameters['test_batch_size'] = 8
    elif parameters['image_resolution'] == 128:
        parameters['train_batch_size'] = 23
        parameters['test_batch_size'] = 23
    elif parameters['image_resolution'] == 64:
        parameters['train_batch_size'] = 70
        parameters['test_batch_size'] = 70
parameters['FT_radial_sampling'] = 4
parameters['predicted_frame'] = 'middle'
parameters['num_coils'] = 8
parameters['dataloader_num_workers'] = 0
parameters['optimizer'] = 'Adam'
parameters['scheduler'] = 'StepLR'
parameters['memoise_disable'] = False
parameters['image_space_real'] = False
parameters['optimizer_params'] = (0.9, 0.999)
parameters['scheduler_params'] = {
    'base_lr': 3e-4,
    'max_lr': 1e-3,
    'step_size_up': 10,
    'mode': 'triangular',
    'step_size': parameters['num_epochs']//3,
    'gamma': 0.5,
    'verbose': True
}
parameters['loss_recon'] = 'L2'
parameters['loss_FT'] = 'None'
parameters['loss_reconstructed_FT'] = 'None'
parameters['beta1'] = 1
parameters['beta2'] = 0.5
parameters['Automatic_Mixed_Precision'] = False
parameters['loss_params'] = {
    'SSIM_window': 11,
    'alpha_phase': 1,
    'alpha_amp': 1,
    'grayscale': True,
    'deterministic': False,
    'watson_pretrained': True,
}