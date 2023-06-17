parameters = {}
parameters['image_resolution'] = 256
parameters['train_batch_size'] = 2
parameters['test_batch_size'] = 2
parameters['lr_kspace'] = 3e-4
parameters['lr_ispace'] = 1e-5
parameters['num_epochs_ispace'] = 0
parameters['num_epochs_kspace'] = 1000
parameters['kspace_architecture'] = 'KLSTM1'
parameters['ispace_architecture'] = 'ILSTM1'
parameters['history_length'] = 0
parameters['loop_videos'] = 30
parameters['dataset'] = 'acdc'
parameters['train_test_split'] = 0.8
parameters['normalisation'] = False
parameters['window_size'] = -1
parameters['init_skip_frames'] = 10
parameters['SHM_looping'] = False
parameters['FT_radial_sampling'] = 20
parameters['num_coils'] = 8
parameters['scale_input_fft'] = False
parameters['dataloader_num_workers'] = 0
parameters['optimizer'] = 'Adam'
parameters['scheduler'] = 'StepLR'
parameters['optimizer_params'] = (0.9, 0.999)
parameters['scheduler_params'] = {
    'base_lr': 3e-4,
    'max_lr': 1e-3,
    'step_size_up': 10,
    'mode': 'triangular',
    'step_size': parameters['num_epochs_kspace']//3,
    'gamma': 0.5,
    'verbose': True
}
parameters['Automatic_Mixed_Precision'] = False
parameters['predicted_frame'] = 'last'
parameters['loss_params'] = {
    'SSIM_window': 11,
    'alpha_phase': 1,
    'alpha_amp': 1,
    'grayscale': True,
    'deterministic': False,
    'watson_pretrained': True,
}
parameters['NUFFT_numpoints'] = 8
parameters['NUFFT_kbwidth'] = 0.84
parameters['shuffle_coils'] = True
parameters['memoise'] = True