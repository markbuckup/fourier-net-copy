import numpy as np
parameters = {}

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
##### These Parameters will mostly be unchanged
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

parameters['save_folder'] = '/Data/ContijochLab/projects/cineMRIRecon'
parameters['image_resolution'] = 256
parameters['kspace_architecture'] = 'MDCNN'
assert(parameters['kspace_architecture'] in ['KSpace_RNN', 'MDCNN'])
parameters['ispace_architecture'] = 'Identity'
if parameters['kspace_architecture'] == 'MDCNN':
    assert(parameters['ispace_architecture'] == 'Identity')
parameters['dataset'] = 'acdc'

########################################################################################################################################################



########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
##### Training Parameters
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
parameters['train_batch_size'] = 1
parameters['test_batch_size'] = 1
parameters['lr_kspace'] = 1e-5
parameters['lr_ispace'] = 1e-5
#Total Number of Epochs
parameters['num_epochs_total'] = 400
# Out of the total epochs, we first train the recurrent subnetwork as follows:
# We train the k-space RNN for epochs = parameters['num_epochs_recurrent']
# We train the image LSTM for epochs = parameters['num_epochs_ilstm']
# We train both the kspace RNN and the image LSTM for epochs = parameters['num_epochs_windowed']
#           But we train the image LSTM for the last 'num_epochs_ilstm' epochs (refer to the example)
#           We also train the windowed mode for the last 'num_epochs_windowed' epochs (refer to the example)
# Assume the following 
#      parameters['num_epochs_total'] = 600
#      parameters['num_epochs_recurrent'] = 300
#      parameters['num_epochs_ilstm'] = 200
#      parameters['num_epochs_windowed'] = 100
#      parameters['num_epochs_unet'] = 300
# Epochs 000-100      ->  K-Space RNN Training
# Epochs 100-200      ->  K-Space RNN + Image LSTM Training
# Epochs 200-300      ->  K-Space RNN + Image LSTM Training in windowed mode
# Epochs 300-600      ->  UNet Training
parameters['num_epochs_recurrent'] = 400
parameters['num_epochs_ilstm'] = 0
parameters['num_epochs_windowed'] = 0
parameters['num_epochs_unet'] = 0

assert(parameters['num_epochs_ilstm'] < parameters['num_epochs_recurrent'])
assert(parameters['num_epochs_windowed'] < parameters['num_epochs_recurrent'])
assert(parameters['num_epochs_recurrent'] <= parameters['num_epochs_total'])
assert(parameters['num_epochs_unet'] <= parameters['num_epochs_total'])

# Choice of optimizer and scheduler - Don't change these
parameters['optimizer'] = 'Adam'
parameters['scheduler'] = 'CyclicLR'
parameters['scheduler_params'] = {
    'base_lr': 4e-6,
    'max_lr': 4e-4,
    'step_size_up': 3000,
    'mode': 'exp_range',
    'step_size': parameters['num_epochs_recurrent']//3,
    'gamma': 0.9999,
    'verbose': True,
    'cycle_momentum': False,
}
parameters['ispace_scheduler_params'] = {
    'base_lr': 4e-6,
    'max_lr': 4e-4,
    'step_size_up': 400,
    'mode': 'exp_range',
    'step_size': parameters['num_epochs_unet']//3,
    'gamma': 0.9999,
    'verbose': True,
    'cycle_momentum': False,
}
assert(parameters['optimizer']) in ['Adam', 'SGD']
assert(parameters['scheduler']) in ['StepLR', 'None', 'CyclicLR']
assert(parameters['scheduler_params']['mode']) in ['triangular', 'triangular2', 'exp_range']

########################################################################################################################################################



########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
##### Architecture Parameters
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

# Logarithm base for the log of magnitude values
parameters['logarithm_base'] = 10
# K-space RNN gates have skip connections
# Undersampled input from the MRI machine is concatenated after every cnn layer to ensure actual data is preserved | Preferred True
parameters['kspace_rnn_skip_connections'] = True

# Coil combination performed by a UNet or Sum of Squares | Preferred UNet
parameters['coil_combine'] = 'UNET'
assert(parameters['coil_combine'] in ['SOS', 'UNET'])

# The K-space RNN input is a concatenation of the undersampled data and the mask of the locations of newly acquired spokes. 
# This parameter enables/disables the masks as input | Preferred True
parameters['rnn_input_mask'] = True

# Number of layers in the K-space RNN gates
parameters['n_layers'] = 3

# Number of intermediate (hidden) channels in the K-space RNN gates
parameters['n_hidden'] = 16

# Number of kspace RNN cells - you can have multiple kspace RNNs coupled one after the other [Do not change]
parameters['n_rnn_cells'] = 1

# [Do not change any of these]
# forget_gate_coupled applies the following constraint - forget gate = 1-input_gate | Preferred True
parameters['forget_gate_coupled'] = True
# forget_gate_coupled applies the following constraint - All coils have the same forget gate | Preferred True
parameters['forget_gate_same_coils'] = True
# forget_gate_coupled applies the following constraint - Both phase and magnitude have the same forget gate | Preferred True
parameters['forget_gate_same_phase_mag'] = True
# The kspace RNN and the image lstm will work in a coil-wise fashion | Preferred True
parameters['coilwise'] = True

# Window size - Do not remember frames before this window in the k-space RNN Cell state
# np.inf for no window
# Even when the window size is finite, the training might not happen for the first few epochs according to the number of epochs
parameters['window_size'] = [np.inf]

# Disables/Enables the image lstm | Preferred True
parameters['image_lstm'] = False
parameters['unet_instead_of_ilstm'] = False
if parameters['unet_instead_of_ilstm']:
    assert(parameters['image_lstm'])


# LSTMs usually concatenate the previous predictions as input to every forward pass
# Keep these False by default to avoid lag
parameters['gate_cat_prev_output'] = False
parameters['ilstm_gate_cat_prev_output'] = False

# only for ABLATION studies - will skip the kspace rnn
parameters['skip_kspace_rnn'] = False

# The output should be real/complex in case we need to predict the phase
# NOTE, final prediction complex is not yet supported - Do not use without extensive testing.
parameters['final_prediction_real'] = True
assert(parameters['final_prediction_real'])

########################################################################################################################################################




########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
##### Dataset parameters
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

# Mini version of the ACDC dataset - using only one slice of every patient
parameters['acdc_debug_mini'] = False
# Memoises the recurrent network outputs for a fast UNet training
# avoids repetetive forwards passes of the recurrent subnetwork 
# the UNet and the K-space RNN are not trained together
parameters['memoise_ispace'] = True

# Number of frames used for training the K-space RNN - purely based on GPU contraints
parameters['loop_videos'] = 30
# If history length is k, the appends k frames from the previous cardiac cycles at the same phase to the input (as channels)
parameters['history_length'] = 0

parameters['train_test_split'] = 0.8
# The preprocessed dataset must exist for these coils and num_spokes combination
parameters['kspace_num_spokes'] = 10
parameters['num_coils'] = 8

# 6/8 is a sweet spot
parameters['dataloader_num_workers'] = 8


########################################################################################################################################################


########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
##### Loss Parameters
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################


# Applies a gaussian weight to the real predictions
# Higher weight for the center - should reduce the lag!
parameters['center_weighted_loss'] = False
parameters['lstm_forget_gate_loss'] = True
parameters['lstm_input_gate_loss'] = True

# Since the initial prediction by RNNs is very bad, we skip the first few frames from the loss
parameters['init_skip_frames'] = 8

parameters['loss_params'] = {
    'SSIM_window': 11,
    'alpha_phase': 1,
    'alpha_amp': 1,
    'grayscale': True,
    'deterministic': False,
    'watson_pretrained': True,
}

########################################################################################################################################################


########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
##### MDCNN Architecture Parameters
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################


parameters['window_size'] = 7
parameters['image_space_real'] = False


########################################################################################################################################################