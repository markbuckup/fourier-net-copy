parameters = {}
parameters['train_batch_size'] = 40
parameters['test_batch_size'] = 40
parameters['lr'] = 1e-3
parameters['num_epochs'] = 50
parameters['train_test_split'] = 0.8
parameters['normalisation'] = False
parameters['image_resolution'] = 64
parameters['window_size'] = 7
parameters['FT_radial_sampling'] = 112
parameters['predicted_frame'] = 'middle'
parameters['num_coils'] = 8
parameters['dataloader_num_workers'] = 0
parameters['optimizer'] = 'Adam'
parameters['scheduler'] = 'StepLR'
parameters['optimizer_params'] = (0.9, 0.999)
parameters['scheduler_params'] = {
    'base_lr': 3e-4,
    'max_lr': 1e-3,
    'step_size_up': 10,
    'mode': 'triangular',
    'step_size': parameters['num_epochs']//3,
    'gamma': 0.1,
    'verbose': True
}
parameters['loss_recon'] = 'L1'
parameters['loss_FT'] = 'None'
parameters['loss_reconstructed_FT'] = 'None'
parameters['train_losses'] = []
parameters['test_losses'] = []
parameters['beta1'] = 1
parameters['beta2'] = 0.5
# parameters['Automatic_Mixed_Precision'] = False
parameters['loss_params'] = {
    'SSIM_window': 11,
    'alpha_phase': 1,
    'alpha_amp': 1,
    'grayscale': True,
    'deterministic': False,
    'watson_pretrained': True,
}