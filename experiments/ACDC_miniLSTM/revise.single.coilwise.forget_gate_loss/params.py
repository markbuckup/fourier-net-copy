parameters = {}
parameters['save_folder'] = '/Data/ContijochLab/projects/cineMRIRecon'
parameters['image_resolution'] = 256
parameters['train_batch_size'] = 1
parameters['test_batch_size'] = 1
parameters['lr_kspace'] = 1e-5
parameters['lr_ispace'] = 1e-5
parameters['num_epochs_ispace'] = 200
parameters['num_epochs_kspace'] = 200
parameters['num_epochs_total'] = 400
assert(parameters['num_epochs_kspace'] <= parameters['num_epochs_total'])
assert(parameters['num_epochs_ispace'] <= parameters['num_epochs_total'])
parameters['kspace_architecture'] = 'KLSTM1'
parameters['double_kspace_proc'] = False
parameters['kspace_combine_coils'] = False
parameters['end-to-end-supervision'] = False
parameters['kspace_real_loss_only'] = False

parameters['lstm_input_mask'] = True
parameters['concat'] = True
parameters['n_layers'] = 4
parameters['n_hidden'] = 16
parameters['n_lstm_cells'] = 1
parameters['forget_gate_coupled'] = True
parameters['forget_gate_same_coils'] = True
parameters['forget_gate_same_phase_mag'] = True
parameters['logarithm_base'] = 10

parameters['skip_kspace_lstm'] = False
parameters['coilwise'] = True
assert( not (parameters['coilwise'] and parameters['kspace_combine_coils']))
parameters['crop_loss'] = False
parameters['lstm_input_proc_identity'] = False
parameters['lstm_forget_gate_loss'] = True
parameters['lstm_input_gate_loss'] = False
parameters['coil_combine'] = 'SOS'
assert(parameters['coil_combine'] in ['SOS', 'UNET'])



parameters['ispace_lstm'] = False
parameters['ispace_architecture'] = 'ILSTM1'
parameters['image_space_real'] = True
parameters['history_length'] = 0
parameters['loop_videos'] = 30
parameters['dataset'] = 'acdc'
parameters['train_test_split'] = 0.8
parameters['normalisation'] = False
parameters['window_size'] = -1
parameters['init_skip_frames'] = 20
parameters['SHM_looping'] = False
parameters['FT_radial_sampling'] = 10
parameters['num_coils'] = 8
parameters['scale_input_fft'] = False
parameters['dataloader_num_workers'] = 6
parameters['optimizer'] = 'Adam'
parameters['scheduler'] = 'CyclicLR'
parameters['optimizer_params'] = (0.9, 0.999)
parameters['scheduler_params'] = {
    'base_lr': 4e-6,
    'max_lr': 4e-4,
    'step_size_up': 600,
    'mode': 'exp_range',
    'step_size': parameters['num_epochs_kspace']//3,
    'gamma': 0.9999,
    'verbose': True,
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
parameters['shuffle_coils'] = False
parameters['memoise'] = False
parameters['memoise_RAM'] = False

parameters['kspace_predict_mode'] = 'thetas'
# parameters['kspace_predict_mode'] = 'thetas' or 'cosine' or 'unit-vector'
parameters['loss_phase'] = 'raw_L1'
# parameters['loss_phase'] = 'L1' or 'Cosine' or 'raw_L1'
parameters['kspace_tanh'] = False
parameters['ground_truth_weight'] = 1
parameters['ground_truth_enforce'] = False