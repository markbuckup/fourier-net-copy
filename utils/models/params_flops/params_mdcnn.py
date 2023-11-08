parameters = {}
parameters['image_resolution'] = 256
parameters['train_batch_size'] = 15
parameters['test_batch_size'] = 12
parameters['lr'] = 1e-3
parameters['num_epochs'] = 250
parameters['image_space_real'] = True
parameters['loop_videos'] = 30
parameters['dataset'] = 'acdc'
parameters['train_test_split'] = 0.8
parameters['normalisation'] = False
parameters['window_size'] = 7
parameters['init_skip_frames'] = parameters['window_size']-1
parameters['SHM_looping'] = False
parameters['FT_radial_sampling'] = 10
parameters['num_coils'] = 8
parameters['scale_input_fft'] = False
parameters['dataloader_num_workers'] = 2
parameters['optimizer'] = 'Adam'
parameters['scheduler'] = 'CyclicLR'
parameters['optimizer_params'] = (0.9, 0.999)
parameters['scheduler_params'] = {
    'base_lr': 4e-5,
    'max_lr': 1e-4,
    'step_size_up': 10,
    'mode': 'triangular',
    'step_size': parameters['num_epochs']//3,
    'gamma': 0.5,
    'verbose': True
}
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
parameters['shuffle_coils'] = False
parameters['memoise'] = False
parameters['memoise_RAM'] = False