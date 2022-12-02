parameters = {}
parameters['image_resolution'] = 64
parameters['train_batch_size'] = 15
parameters['test_batch_size'] = 15
parameters['lr_kspace'] = 1e-5
parameters['lr_ispace'] = 1e-4
parameters['lr_gate'] = 3e-4
parameters['num_epochs'] = 50
parameters['architecture'] = 'MDCNNGRU2'
parameters['dataset'] = 'acdc'
parameters['train_test_split'] = 0.8
parameters['normalisation'] = False
parameters['window_size'] = 7
parameters['loop_videos'] = 60
parameters['mini_batch_length'] = 30
parameters['init_skip_frames'] = 10
parameters['SHM_looping'] = True
parameters['image_space_real'] = True
parameters['FT_radial_sampling'] = 112
parameters['predicted_frame'] = 'middle'
parameters['num_coils'] = 8
parameters['dataloader_num_workers'] = 0
parameters['optimizer'] = 'Adam'
parameters['scheduler'] = 'StepLR'
parameters['memoise_disable'] = False
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