import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms
import numpy as np
import os
import sys
import time
import kornia
import matplotlib.pyplot as plt
import utils.models.complexCNNs.cmplx_conv as cmplx_conv
import utils.models.complexCNNs.cmplx_dropout as cmplx_dropout
import utils.models.complexCNNs.cmplx_upsample as cmplx_upsample
import utils.models.complexCNNs.cmplx_activation as cmplx_activation
import utils.models.complexCNNs.radial_bn as radial_bn

import sys
sys.path.append('../../')

from utils.functions import fetch_loss_function

EPS = 1e-10
CEPS = torch.complex(torch.tensor(EPS),torch.tensor(EPS))

class Identity(nn.Module):
    """
    A simple identity module - placeholder for absent modules during ablation studies

    Attributes
    ------------
        n_rnn_cells : int
            Place holder argument
        m : nn.Module
            Place holder attribute so that the module has parameters.
    ================================================================================================        
    """

    def __init__(self, n_rnn_cells = 1, image_lstm = False):   # AERS: Added param_dic because sphinx needed it
        super(Identity, self).__init__()
        if not image_lstm:
            self.m = nn.Linear(3,3)
        self.n_rnn_cells = n_rnn_cells
        
    
    def forward(self, hist_mag, hist_phase, gt_mask = None, mag_prev_outputs = None, phase_prev_outputs = None):
        """
        Place holder Forward pass - **will never be used**

        Parameters:
        -------------
        - hist_mag : Tensor
            Historical magnitude data.
        - hist_phase : Tensor
            Historical phase data.
        - gt_mask : Tensor, optional
            Ground truth mask.
        - mag_prev_outputs : Tensor, optional
            Previous magnitude outputs.
        - phase_prev_outputs : Tensor, optional
            Previous phase outputs.

        Returns:
        -------------
        - Tuple: List[Tensor], List[Tensor]
            The same outputs as input

        ================================================================================================
        """
        new_mag_outputs = [hist_mag for i in range(self.n_rnn_cells)]
        new_phase_outputs = [hist_phase for i in range(self.n_rnn_cells)]
        
        return new_mag_outputs, new_phase_outputs

class Identity_param(nn.Module):
    """
    A identity module - placeholder for absent modules during ablation studies

    Attributes:
    ------------
    - m : nn.Module
        Linear transformation layer.

    ================================================================================================
    """

    def __init__(self, parameters, proc_device):
        """
        AERS:
        Initialize the Identity_param module.

        This method initializes an instance of the Identity_param module, setting up the internal 
        components and parameters. It includes a placeholder linear layer as an example.

        Parameters:
        ------------
        - parameters (dict): 
            A dictionary containing the configuration parameters for the module. 
            The specific contents and structure of this dictionary depend on the use case.
        - proc_device (str): 
            The device on which the module's computations will be performed. Typically, this 
            is either 'cpu' or 'cuda' to specify the processing device.

        ================================================================================================
        """
        super(Identity_param, self).__init__()
        self.m = nn.Linear(3,3)
        
    def forward(self, x):
        """
        Placeholder Forward pass of the Identity_param module.

        Parameters:
        ------------
        - x : Tensor
            Input tensor.

        Returns:
        ------------
        - x : Tensor

        ================================================================================================
        """
        return x


def mylog(x,base = 10):
    """
    Computes the logarithm of a tensor with a specified base.

    Parameters:
    ------------
    - x : Tensor
        Input tensor.
    - base :int, optional
        Logarithm base. Defaults to 10.

    Returns:
    ------------
        Tensor: Logarithm of the input tensor.

    ================================================================================================
    """
    return x.log10()/torch.tensor(base).log10()

def gaussian_2d(shape, sigma=None):
    """
    Generate a 2D Gaussian mask.

    Parameters:
    ------------
    - shape : tuple
        Shape of the output array (height, width).
    - sigma : float
        Standard deviation of the Gaussian distribution.

    Returns:
    ------------
    - numpy.ndarray: 2D Gaussian mask.

    ================================================================================================
    """
    if sigma is None:
        sigma = shape[0]//5
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sum_h = h.sum()
    if sum_h != 0:
        h /= sum_h
    h = h - h.min()
    h = h/(h.max()+1e-10)
    h *= 9
    h += 1
    return h

def special_trim(x, l = 5, u = 95):
    """
    Trims the values of a tensor to be within the specified percentiles.

    Parameters:
    ------------
    - x : Tensor
        Input tensor.
    - l : int, optional
        Lower percentile. Defaults to 5.
    - u : int, optional
        Upper percentile. Defaults to 95.

    Returns:
    ------------
    - Tensor : Trimmed tensor.
    """
    percentile_95 = np.percentile(x.detach().cpu(), u)
    percentile_5 = np.percentile(x.detach().cpu(), l)
    x = x.clip(percentile_5, percentile_95)
    return x

def fetch_models(parameters):
    """
    Fetches the LSTM model types based on the given parameters.

    Parameters:
    ------------
    - parameters : dict
        Dictionary of parameters.

    Returns:
    ------------
    - Tuple[Type[nn.Module], Type[nn.Module]]: LSTM model types for image and k-space.

    ================================================================================================
    """
    ispace_name = parameters['ispace_architecture']
    kspace_name = parameters['kspace_architecture']
    if ispace_name == 'ILSTM1':
        im = ImageSpaceModel1
    elif ispace_name == 'Identity':
        im = Identity_param

    if kspace_name == 'KSpace_RNN':
        km = convLSTM_Kspace1
    if kspace_name == 'MDCNN':
        from utils.models.MDCNN import MDCNN
        km = MDCNN
    return km, im

class concatConv(nn.Module):
    """
    Concatenated convolutional layers module.

    Attributes:
    ------------
    - layerlist : nn.ModuleList
        List of convolutional layers.
    - skip_connections : bool
        Flag for using skip connections.
    - relu_func : nn.Module
        ReLU activation function.
    - n_layers : int
        Number of layers.

    ================================================================================================
    """
    def __init__(self, cnn_func, relu_func, gate_input_size = 8, hidden_channels = 32, gate_output_size = 1, n_layers = 4, skip_connections = True):
        """
        Initializes the concatConv module.

        Parameters:
        ------------
        - cnn_func : Type[nn.Module]
            Convolution function - real or complex.
        - relu_func : Type[nn.Module]
            ReLU function.
        - gate_input_size : int, optional
            Input size of the gate. 
                Defaults to 8.
        - hidden_channels : int, optional
            Number of hidden channels. 
                Defaults to 32.
        - gate_output_size : int, optional
            Output size of the gate. 
                Defaults to 1.
        - n_layers : int, optional
            Number of layers. 
                Defaults to 4.
        - skip_connections : bool, optional
            Flag for using skip connections. 
                Defaults to True.

        ================================================================================================
        """
        super(concatConv, self).__init__()
        self.layerlist = []
        self.skip_connections = skip_connections
        self.relu_func = relu_func()
        self.n_layers = n_layers
        if self.n_layers == 1:
            self.layerlist.append(cnn_func(gate_input_size, gate_output_size, (1,1), stride = (1,1), padding = (0,0)))
        else:
            self.layerlist.append(cnn_func(gate_input_size, hidden_channels, (3,3), stride = (1,1), padding = (1,1)))
            inlen = hidden_channels
            if self.skip_connections:
                skiplen = gate_input_size
            else:
                skiplen = 0
            for i in range(n_layers-2):
                self.layerlist.append(cnn_func(inlen+skiplen, hidden_channels, (3,3), stride = (1,1), padding = (1,1)))
                inlen = hidden_channels
            self.layerlist.append(cnn_func(inlen+skiplen, gate_output_size, (1,1), stride = (1,1), padding = (0,0)))

        self.layerlist = nn.ModuleList(self.layerlist)

    def forward(self, x):
        """
        Forward pass of the concatConv module.

        Parameters:
        ------------
        - x : Tensor
            Input tensor.

        Returns:
        ------------
        - Output tensor : Tensor
            
        ================================================================================================
        """
        if self.n_layers == 1:
            return self.layerlist[0](x)

        to_cat = x.clone()
        curr_output = self.layerlist[0](x)
        
        for i,layer in enumerate(self.layerlist[1:self.n_layers-1]):
            if self.skip_connections:
                input = torch.cat((curr_output, to_cat), 1)
            else:
                input = curr_output
            curr_output = self.relu_func(layer(input))

        if self.skip_connections:
            input = torch.cat((curr_output, to_cat), 1)
        else:
            input = curr_output
        return self.layerlist[-1](input)


# AERS: k-space RNN
class RecurrentModule(nn.Module):
    """
    The Recurrent Module for undersampled k-space data processing. Contains the kspace-RNN and the image LSTM.

    Attributes:
    -----------
        - Various attributes for LSTM cell configuration.

    ================================================================================================
    """
    def __init__(self, history_length = 0, num_coils = 8, forget_gate_coupled = False, forget_gate_same_coils = False, forget_gate_same_phase_mag = False, rnn_input_mask = True, skip_connections = False, n_layers = 3, n_hidden = 12, n_rnn_cells = 1, coilwise = True, gate_cat_prev_output = False):
        """
        Initializes the convLSTMcell_kspace module.

        Parameters:
        ------------
        - history_length : int, optional
            Length of the history to append to the undersampled input. Appends historical frames from previous cardiac cycles. 
                Defaults to 0.
        - num_coils : int, optional
            Number of coils. 
                Defaults to 8.
        - forget_gate_coupled : bool, optional
            Flag for coupled forget gate - forget gate mask and input gate mask sum to 1. 
                Defaults to False.
        - forget_gate_same_coils : bool, optional
            Flag for same forget gate for all coils. 
                Defaults to False.
        - forget_gate_same_phase_mag : bool, optional
            Flag for same forget gate for phase and magnitude. 
                Defaults to False.
        - rnn_input_mask : bool, optional
            Flag for appending the mask of the locations of newly acquired data to the RNN input. 
                Defaults to True.
        - skip_connections : bool, optional
            Flag for having skip connections. 
                Defaults to False.
        - n_layers : int, optional
            Number of layers in the K-space RNN gates. 
                Defaults to 3.
        - n_hidden :int, optional
            Number of channels in the K-space RNN gates. 
                Defaults to 12.
        - n_rnn_cells : int, optional
            Number of RNN cells - can be coupled one after the other. 
                Defaults to 1.
        - coilwise : bool, optional
            If enabled, each coil of the input is processed independently. 
                Defaults to True.
        - gate_cat_prev_output : bool, optional
            Flag for appending the previously predicted frame to the RNN input. 
                Defaults to True.
        
        ================================================================================================
        """
        super(RecurrentModule, self).__init__()
        self.n_rnn_cells = n_rnn_cells
        self.n_hidden = n_hidden
        self.history_length = history_length
        self.num_coils = num_coils
        self.skip_connections = skip_connections
        self.coilwise = coilwise
        self.forget_gate_coupled = forget_gate_coupled
        self.forget_gate_same_coils = forget_gate_same_coils
        self.forget_gate_same_phase_mag = forget_gate_same_phase_mag
        self.rnn_input_mask = rnn_input_mask
        self.gate_cat_prev_output = gate_cat_prev_output
        
        mag_cnn_func = nn.Conv2d
        mag_relu_func = nn.LeakyReLU
        
        phase_cnn_func = nn.Conv2d
        phase_relu_func = nn.LeakyReLU

        self.phase_activation = lambda x: x
        self.input_gate_output_size = self.num_coils

        if self.coilwise:
            self.input_gate_output_size = 1
            gate_input_size = 1 + ((self.history_length))
            if self.gate_cat_prev_output:
                gate_input_size += 1
        else:
            gate_input_size = ((self.history_length)*self.num_coils)
            if self.gate_cat_prev_output:
                gate_input_size += self.num_coils

        if self.rnn_input_mask:
            gate_input_size += 1

        hidden_channels = self.n_hidden

        if self.forget_gate_same_coils:
            forget_gate_output_size = 1
        else:
            forget_gate_output_size = self.num_coils

        # AERS: This is not describe in the thesis document or presentation (Video June 10th, Part 3,  55:20  min)
        # AERS: In order for the mag and phase to match, the forget masks of both need to be the same. To achieve this, there is a common NN that forgets the forget mask for both phase and mag.
        #       Those are imposed per coil, regardless of their location.
                
    # AERS: Gates definition
        # AERS: Input gate for the magnitude. It uses a concatConv class (from LSTM cell, see Definition) because it has three layers of convolutions. This allows control over the number of layers (it adds convolutional layers and ReLus),
        # while concatenating the gt to each layer and avoid losing information across layers. NRM: It is actually adding the values in the end.
        self.mag_inputGates = nn.ModuleList([concatConv(mag_cnn_func, mag_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, skip_connections = self.skip_connections) for i in range(self.n_rnn_cells)])
        if not self.forget_gate_same_phase_mag:
            self.phase_inputGates = nn.ModuleList([concatConv(phase_cnn_func, phase_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, skip_connections = self.skip_connections) for i in range(self.n_rnn_cells)])
        if not self.forget_gate_coupled:
            self.mag_forgetGates = nn.ModuleList([concatConv(mag_cnn_func, mag_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, skip_connections = self.skip_connections) for i in range(self.n_rnn_cells)])
            if not self.forget_gate_same_phase_mag:
                self.phase_forgetGates = nn.ModuleList([concatConv(phase_cnn_func, phase_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, skip_connections = self.skip_connections) for i in range(self.n_rnn_cells)])
        self.mag_outputGates = nn.ModuleList([concatConv(mag_cnn_func, mag_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, skip_connections = self.skip_connections) for i in range(self.n_rnn_cells)])
        self.phase_outputGates = nn.ModuleList([concatConv(phase_cnn_func, phase_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, skip_connections = self.skip_connections) for i in range(self.n_rnn_cells)])
        self.mag_inputProcs = nn.ModuleList([concatConv(mag_cnn_func, mag_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, skip_connections = self.skip_connections) for i in range(self.n_rnn_cells)])
        self.phase_inputProcs = nn.ModuleList([concatConv(phase_cnn_func, phase_relu_func, gate_input_size, hidden_channels, forget_gate_output_size, n_layers = n_layers, skip_connections = self.skip_connections) for i in range(self.n_rnn_cells)])
        
    # AERS: Forward pass of the k-space RNN
    def forward(self, hist_mag, hist_phase, background = None, gt_mask = None, mag_prev_outputs = None, phase_prev_outputs = None, window_size = np.inf, mag_gates_remember = None, phase_gates_remember = None, eval = False):
        """
        Forward pass of the RecurrentModule.

        Parameters:
        ------------
        - hist_mag : Tensor
            Log Magnitude data. If history is provided, then this will have multiple frames appended as channnels.
        - hist_phase : Tensor
            Phase data as angles. If history is provided, then this will have multiple frames appended as channnels.
        - background : Tensor, optional
            Background mask. 
                Defaults to None.
        - gt_mask : Tensor, optional
            Ground truth mask. 
                Defaults to None.
        - mag_prev_outputs : Tensor, optional
            Previous magnitude outputs. 
                Defaults to None.
        -  phase_prev_outputs : Tensor, optional
            Previous phase outputs. 
                Defaults to None.
        - window_size : int, optional
            Window size for gating. 
                Defaults to np.inf.
        - mag_gates_remember : list, optional
            List to remember magnitude gates to enforce windowing. 
                Defaults to None.
        - phase_gates_remember : list, optional
            List to remember phase gates to enforce windowing. 
                Defaults to None.
        - eval : bool, optional
            Evaluation flag - if yes, then do not compute loss. 
                Defaults to False.

        Returns:
        ----------
        - [New magnitude and phase outputs, loss for forget gate, loss for input gate, remembered magnitude gates, remembered phase gates] : Tuple[List[Tensor], List[Tensor], Tensor, Tensor, List, List]
            
        ================================================================================================
        """
        del gt_mask
        if background is None:
            background = torch.ones_like(hist_mag) == 1
        foreground = torch.logical_not(background).float().to(hist_mag.device)

        # AERS: If mag_prev_outputs is None, this is the first frame and everything is initialized to zeros
        if mag_prev_outputs is None:
            if self.coilwise:
                mag_shape1 = (hist_mag.shape[0], self.num_coils, *hist_mag.shape[2:])
                phase_shape1 = (hist_phase.shape[0], self.num_coils, *hist_phase.shape[2:])
            else:
                mag_shape1 = (hist_mag.shape[0], self.input_gate_output_size, *hist_mag.shape[2:])
                phase_shape1 = (hist_phase.shape[0], self.input_gate_output_size, *hist_phase.shape[2:])
            mag_prev_outputs = [torch.zeros(mag_shape1, device = hist_mag.device) for _ in range(self.n_rnn_cells)]
            phase_prev_outputs = [torch.zeros(phase_shape1, device = hist_phase.device) for _ in range(self.n_rnn_cells)]


        # AERS: Everything gets reshaped from Batch,Channels to Batch*Channels
        og_B, og_C, _,_ = hist_mag.shape
        if self.coilwise:
            hist_mag = hist_mag.reshape(og_B*og_C, 1, *hist_mag.shape[2:])
            hist_phase = hist_phase.reshape(og_B*og_C, 1, *hist_phase.shape[2:])
            foreground = foreground.reshape(og_B*og_C, 1, *hist_phase.shape[2:])
            # gt_mask = gt_mask.repeat((1, self.num_coils,1,1)).reshape(og_B*og_C, 1, *hist_phase.shape[2:])
            for i_cell in range(self.n_rnn_cells):
                mag_prev_outputs[i_cell] = mag_prev_outputs[i_cell].reshape(og_B*og_C, 1, *mag_prev_outputs[i_cell].shape[2:])
                phase_prev_outputs[i_cell] = phase_prev_outputs[i_cell].reshape(og_B*og_C, 1, *phase_prev_outputs[i_cell].shape[2:])

        new_mag_outputs = [hist_mag]
        new_phase_outputs = [hist_phase]
        loss_forget_gate = 0
        loss_input_gate = 0
        criterionL1 = nn.L1Loss()
        criterionL2 = nn.MSELoss()
        if mag_gates_remember is None:
            mag_gates_remember = [[] for i in range(self.n_rnn_cells)]
            if not self.forget_gate_same_phase_mag:
                phase_gates_remember = [[] for i in range(self.n_rnn_cells)]

        # AERS:
        # mag_inp_cat: magnitude input concatenated
        # foreground: mask

        for i_cell in range(self.n_rnn_cells):
            if self.rnn_input_mask:             # AERS: Input concatenation. Niraj tested not concatenating that masks, but we do need them. 
                if self.gate_cat_prev_output:
                    mag_inp_cat = torch.cat((new_mag_outputs[i_cell], hist_mag, foreground), 1)
                    phase_inp_cat = torch.cat((new_phase_outputs[i_cell], hist_phase, foreground), 1)
                else:
                    mag_inp_cat = torch.cat((hist_mag, foreground), 1)
                    phase_inp_cat = torch.cat((hist_phase, foreground), 1)
            else:
                if self.gate_cat_prev_output:
                    mag_inp_cat = torch.cat((new_mag_outputs[i_cell], hist_mag), 1)
                    phase_inp_cat = torch.cat((new_phase_outputs[i_cell], hist_phase), 1)
                else:
                    mag_inp_cat = hist_mag
                    phase_inp_cat = hist_phase

            # AERS: mag_it, mag_ot and phase_ot are computed
            mag_it = torch.sigmoid(self.mag_inputGates[i_cell](mag_inp_cat))   
            mag_ot = torch.sigmoid(self.mag_outputGates[i_cell](mag_inp_cat))
            phase_ot = torch.sigmoid(self.phase_outputGates[i_cell](phase_inp_cat))

            if not self.forget_gate_same_phase_mag:
                phase_it = torch.sigmoid(self.phase_inputGates[i_cell](phase_inp_cat))
            else:
                phase_it = mag_it       # AERS: now, we replace phase_it with mag_it so that they have the same computation

            # AERS: mag_it should be remembered and look more like the foreground (mask)
            loss_forget_gate += criterionL1(mag_it*foreground, foreground)

            # AERS: imposes that all coils have the same forget gate
            if self.forget_gate_same_coils:
                mag_it = mag_it.repeat(1,self.input_gate_output_size,1,1)
                phase_it = phase_it.repeat(1,self.input_gate_output_size,1,1)
            mag_ot = mag_ot.repeat(1,self.input_gate_output_size,1,1)
            phase_ot = phase_ot.repeat(1,self.input_gate_output_size,1,1)

            # AERS: for each forward pass, rememebr the its and remember them for a few frames. After a few frames, delete this information.
            mag_gates_remember[i_cell].append(mag_it.detach().cpu())
            if not self.forget_gate_same_phase_mag:
                phase_gates_remember[i_cell].append(phase_it.detach().cpu())

            if window_size == np.inf:
                mag_ft = 1 - mag_it     # AERS: by definition, forget gate = 1 - input gate. Niraj added this because he actually trained with it, not ft
                phase_ft = 1 - phase_it
            else:                       # AERS: unclear from video (June 1-th, Part 3, 1:10 hr), but seems to be the part that forgets past frames outside the window
                if len(mag_gates_remember[i_cell][-window_size:-1]) >= 1:
                    mag_ft = torch.stack(mag_gates_remember[i_cell][-window_size:-1], -1).max(-1)[0].to(mag_it.device) - 5*mag_it
                    mag_ft = mag_ft.clip(0,1)
                    if not self.forget_gate_same_phase_mag:
                        phase_ft = torch.stack(phase_gates_remember[i_cell][-window_size:-1], -1).max(-1)[0].to(phase_it.device) - 5*phase_it
                        phase_ft = phase_ft.clip(0,1)
                    else:
                        phase_ft = mag_ft
                else:
                    mag_ft = torch.zeros_like(mag_it, device = mag_it.device)
                    phase_ft = torch.zeros_like(phase_it, device = phase_it.device)

            
            mag_Cthat = self.mag_inputProcs[i_cell](mag_inp_cat)
            phase_Cthat = self.phase_activation(self.phase_inputProcs[i_cell](phase_inp_cat))
            if not eval:
                loss_input_gate += criterionL1(mag_Cthat*foreground, hist_mag*foreground)       # AERS: Calculates the input gate loss for phase and mag by comparing locations inside the mask
                loss_input_gate += criterionL1(phase_Cthat*foreground, hist_phase*foreground)

            # AERS: Appends the new states (new state = new output)
            #       Note that phase it/ft = mag it/ft  However, Ct hat is different between mag and phase
            new_mag_outputs.append(mag_ot*((mag_ft * mag_prev_outputs[i_cell]) + (mag_it * mag_Cthat)))
            new_phase_outputs.append(phase_ot*((phase_ft * phase_prev_outputs[i_cell]) + (phase_it * phase_Cthat)))

            if not eval:
                loss_input_gate += criterionL1(new_mag_outputs[-1]*foreground, hist_mag*foreground)
                loss_input_gate += criterionL1(new_phase_outputs[-1]*foreground, hist_phase*foreground)


        if self.coilwise:
            for i_cell in range(self.n_rnn_cells):  # AERS: Reshapes back to original size
                new_mag_outputs[i_cell+1] = new_mag_outputs[i_cell+1].reshape(og_B,og_C,*new_mag_outputs[i_cell+1].shape[2:])
                new_phase_outputs[i_cell+1] = new_phase_outputs[i_cell+1].reshape(og_B,og_C,*new_phase_outputs[i_cell+1].shape[2:])

        return new_mag_outputs[1:], new_phase_outputs[1:], loss_forget_gate, loss_input_gate, mag_gates_remember, phase_gates_remember

# AERS: Image space LSTM. This is very close to the standard LSTM code available elsewhere.
class convLSTMcell(nn.Module): 
    """
    A convolutional LSTM cell module.

    Attributes:
    ------------
    - Various attributes for LSTM cell configuration.

    ================================================================================================
    """
    def __init__(self, in_channels = 1, out_channels = 1, tanh_mode = False, real_mode = False, ilstm_gate_cat_prev_output = False):
        """
        Initializes the convLSTMcell module.

        Parameters:
        ------------
        - in_channels : int, optional
            Number of input channels. 
                Defaults to 1.
        - out_channels : int, optional
            Number of output channels. 
                Defaults to 1.
        - tanh_mode : bool, optional
            Flag for using tanh activation for the output. 
                Defaults to False.
        - real_mode : bool, optional
            Flag for real mode for the layers. 
                Defaults to False.
        - ilstm_gate_cat_prev_output : bool, optional
            Flag for concatenating previous output to gate input. 
                Defaults to False.

        ================================================================================================
        """
        super(convLSTMcell, self).__init__()
        self.tanh_mode = tanh_mode
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.real_mode = real_mode
        self.ilstm_gate_cat_prev_output = ilstm_gate_cat_prev_output

        # AERS: option from parameters: use a complex convolution of an RNN conv
        if real_mode:
            cnn_func = nn.Conv2d
            relu_func = nn.ReLU
        else:
            cnn_func = cmplx_conv.ComplexConv2d
            relu_func = cmplx_activation.CReLU

        if self.tanh_mode:
            self.activation = lambda x: torch.tanh(x)
        else:
            self.activation = lambda x: x

        if self.ilstm_gate_cat_prev_output:
            gate_input_size = self.in_channels + self.out_channels
        else:
            gate_input_size = self.in_channels

        # AERS: Image space LSTM gates
        self.inputGate = nn.Sequential(
                cnn_func(gate_input_size, 2*self.in_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.in_channels, self.out_channels, (1,1), stride = (1,1), padding = (0,0)),
            )
        self.forgetGate = nn.Sequential(
                cnn_func(gate_input_size, 2*self.in_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.in_channels, self.out_channels, (1,1), stride = (1,1), padding = (0,0)),
            )
        self.outputGate = nn.Sequential(
                cnn_func(gate_input_size, 2*self.in_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.in_channels, self.out_channels, (1,1), stride = (1,1), padding = (0,0)),
            )
        # AERS: This is a NN that generates CT hat
        self.inputProc = nn.Sequential(
                cnn_func(gate_input_size, 2*self.in_channels, (3,3), stride = (1,1), padding = (1,1)),
                relu_func(),
                cnn_func(2*self.in_channels, self.out_channels, (1,1), stride = (1,1), padding = (0,0)),
                relu_func(),
            )

    def forward(self, x, prev_state = None, prev_output = None):
        """
        Forward pass of the convLSTMcell module.

        Parameters:
        ------------
        - x : Tensor
            Input tensor.
        - prev_state : Tensor, optional
            Previous cell state. 
                Defaults to None.
        - prev_output : Tensor, optional
            Previous output. 
                Defaults to None.

        Returns:
        ------------
        - Tuple[Tensor, Tensor]: New cell state and output.

        ================================================================================================
        """
        # AERS: If prev_state is None, this is the first frame and it is initialized to zeros 
        if prev_state is None: 
            shape1 = (x.shape[0], self.out_channels, *x.shape[2:])
            prev_state = torch.zeros(shape1, device = x.device)
            if self.ilstm_gate_cat_prev_output:
                prev_output = torch.zeros(shape1, device = x.device)
        if self.ilstm_gate_cat_prev_output:
            inp_cat = torch.cat((x, prev_output), 1)
        else:
            inp_cat = x

        # AERS: predictions are computed with the forget gate, input gate, and output gave and then a sigmoid is applied to them
        ft = torch.sigmoid(self.forgetGate(inp_cat))
        it = torch.sigmoid(self.inputGate(inp_cat))
        ot = torch.sigmoid(self.outputGate(inp_cat))

        # AERS: use those outputs to compute CT hat (intermediate cell state)
        Cthat = self.activation(self.inputProc(inp_cat))   # AERS: the cell route activate in a tanh, but we don't use that here (unclear Video Jun 10th, Part 3, 51 min)
        Ct_new = (ft * prev_state) + (it * Cthat)
        ht = self.activation(Ct_new)*ot
        
        return Ct_new, ht

# AERS: K-space model call in DDP paradigms and LSTM Trainer
class convLSTM_Kspace1(nn.Module):
    def __init__(self, parameters, proc_device, two_cell = False):      # AERS: arguments– parameters dictionary from the experiment file, GPU # (to initialize sall masks used in every forward pass). two_cell is NOT used. It allows for concatenating LSTMs (I think).
        # AERS: 1) Stores paramters:
        super(convLSTM_Kspace1, self).__init__()
        self.param_dic = parameters                                     # AERS: This is not called self.parameters because when the optimizer is defined in the DDP_LSTMTrainer_nufft.py it takes in the model parameters. The model already has a function with that name, so param_dic is used instead.
        self.history_length = self.param_dic['history_length']          # AERS: History length. If we use ARKS, we want to specify how many past frames we want to use. Currently at 0. 
        self.n_coils = self.param_dic['num_coils']


        self.real_mode = True

        # AERS: Two other modes were initially tested in previous versions. They were deleted and only this one remained because it is the only one that worked.
        # if not self.param_dic['skip_kspace_rnn']:   # AERS: Original from Niraj
        if not self.param_dic.get('skip_kspace_rnn', False):     # AERS: Edited because sphinx was having an issue with this line
        
        # AERS: Define the recurrrent model:
            self.kspace_m = RecurrentModule(
                        num_coils = self.n_coils,
                        history_length = self.history_length,
                        forget_gate_coupled = self.param_dic['forget_gate_coupled'], 
                        forget_gate_same_coils = self.param_dic['forget_gate_same_coils'],
                        forget_gate_same_phase_mag = self.param_dic['forget_gate_same_phase_mag'],
                        rnn_input_mask = self.param_dic.get('rnn_input_mask', None),
                        skip_connections = self.param_dic.get('kspace_rnn_skip_connections', None),
                        n_layers = self.param_dic.get('n_layers', 0),
                        n_hidden = self.param_dic.get('n_hidden', 0),
                        n_rnn_cells = self.param_dic.get('n_rnn_cells',0),
                        coilwise = self.param_dic['coilwise'],
                        gate_cat_prev_output = self.param_dic['gate_cat_prev_output'],
                    ) 
            
            # AERS: Edited line 571 from rnn_input_mask = self.param_dic('rnn_input_mask'), because sphinx had issues with it.
            # AERS: Edited line 572 from skip_connections = self.param_dic['kspace_rnn_skip_connections'], because sphinx had issues with it.
            # AERS: Edited line 573 from n_layers = self.param_dic['n_layers'],
            # AERS: Edited line 574 from n_hidden = self.param_dic['n_hidden'],
            # AERS: Edited line 575 from n_rnn_cells = self.param_dic['n_rnn_cells'],

        else:
            #AERS: If you want to skip the recurrent network (ablation study).
            self.kspace_m = Identity(n_rnn_cells = self.param_dic['n_rnn_cells'], image_lstm = self.param_dic['image_lstm'])

        # AERS: 2) If you want an image model (optional), it is defined using either a U-Net or another convLSTM for image space:
        if self.param_dic['image_lstm']:
            if self.param_dic['unet_instead_of_ilstm']:
                self.ispacem = UNet(
                        in_channels = 1, 
                        out_channels = 1, 
                    )
            else:
                self.ispacem = convLSTMcell(            # AERS: convLSTMcell is built specially for image space
                        tanh_mode = False, 
                        real_mode = True, 
                        in_channels = 1, 
                        out_channels = 1, 
                        ilstm_gate_cat_prev_output = self.param_dic['ilstm_gate_cat_prev_output'],
                    )
        # AERS: 3) Define a loss mask:
        self.SSIM = kornia.metrics.SSIM(11)             # AERS: SSIM metric

        if self.param_dic['center_weighted_loss']:
            mask = gaussian_2d((self.param_dic['image_resolution'],self.param_dic['image_resolution'])).reshape(1,1,self.param_dic['image_resolution'],self.param_dic['image_resolution'])

        else:
            mask = np.ones((1,1,self.param_dic['image_resolution'],self.param_dic['image_resolution']))

        # AERS: Loss mask for the real values. Instead of a crop loss, a Gaussian loss mask so that the center weight more, because the periphery doesn't change as much. This makes the model focus more on the center of k-sapce. 
        # AERS: The size of the Gaussian can be changed with the sigma error below (related to image_resolution). It is currently very narrow (per Niraj).
        # AERS: This worked better the the crop loss mask because masking everything ouside the crop box to zero implies that the model can predict anything in the periphery.
        self.lossmask = torch.FloatTensor(mask).to(proc_device)

        mask = torch.FloatTensor(gaussian_2d((self.param_dic['image_resolution'],self.param_dic['image_resolution']), sigma = self.param_dic['image_resolution']//10))
        mask = torch.fft.fftshift(mask) 
        mask = mask - mask.min()
        mask = mask / (mask.max() + EPS)
        mask = (1-mask).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.predr_mask = torch.FloatTensor(mask).to(proc_device)

    def time_analysis(self, fft_exp, device, periods, ispace_model):
        times = []
        with torch.no_grad():
            prev_outputs1 = None
            prev_outputs2 = None
            prev_state3 = None
            prev_output3 = None
            mag_gates_remember = None
            phase_gates_remember = None

            predr = torch.zeros(fft_exp.shape)

            length = fft_exp.shape[-1]
            x_size, y_size = length, length
            x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
            cell = (length//2, length//2)
            dists = ((x_arr - cell[0])**2 + (y_arr - cell[1])**2)**0.22
            dists = torch.FloatTensor(dists+1).to(device).unsqueeze(0).unsqueeze(0)
            if not self.param_dic['center_weighted_loss']:
                dists = torch.ones_like(dists, device = device).unsqueeze(0).unsqueeze(0)
            cycle_mask = ((x_arr - cell[0])**2 + (y_arr - cell[1])**2)**2
            cycle_mask[cycle_mask > cycle_mask[0,x_size//2]] = -1
            cycle_mask[cycle_mask > -1] = 1
            cycle_mask[cycle_mask == -1] = 0
            cycle_mask = torch.FloatTensor(cycle_mask).to(device).unsqueeze(0).unsqueeze(0)

            
            for ti in range(fft_exp.shape[1]):
                start1 = time.time()
                hist_ind = (torch.arange(self.history_length+1).repeat(fft_exp.shape[0],1) - self.history_length)
                hist_ind = hist_ind * periods.reshape(-1,1).cpu()
                hist_ind += ti
                temp1 = hist_ind.clone()
                temp1[temp1 < 0] = 9999999999
                min_vals = temp1.min(1)[0]
                base = torch.zeros(temp1.shape) + min_vals.reshape(-1,1)

                hist_ind = hist_ind - base
                hist_ind[hist_ind < 0] = 0
                hist_ind = (hist_ind + base).long()

                mult = (torch.arange(fft_exp.shape[0])*fft_exp.shape[1]).reshape(-1,1)
                hist_ind = hist_ind + mult
                hist_fft = fft_exp.reshape(-1, *fft_exp.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *fft_exp.shape[3:]).to(device)
                hist_mag = mylog((hist_fft.abs()+EPS), base = self.param_dic['logarithm_base'])
                hist_phase = hist_fft / (self.param_dic['logarithm_base']**hist_mag)
                hist_phase = torch.stack((hist_phase.real, hist_phase.imag), -1)

                background = ((hist_phase[:,:,:,:,1].abs() + hist_phase[:,:,:,:,0].abs()) < 1)
                hist_phase = torch.atan2(hist_phase[:,:,:,:,1]+EPS,hist_phase[:,:,:,:,0]) + 4
                hist_phase[background] = 0
                hist_mag[background] = 0
    
                if prev_outputs2 is not None:
                    prev_outputs2 = [(x + 5) for x in prev_outputs2]
                    prev_outputs1 = [x + 4 for x in prev_outputs1]

                curr_mask = None
                random_window_size = np.random.choice(self.param_dic['window_size'])

                # AERS: This is the forward pass
                prev_outputs2, prev_outputs1, loss_forget_gate_curr, loss_input_gate_curr, mag_gates_remember, phase_gates_remember = self.kspace_m(
                                                                                                                                                        hist_mag,
                                                                                                                                                        hist_phase,
                                                                                                                                                        background = background,
                                                                                                                                                        gt_mask = curr_mask,
                                                                                                                                                        mag_prev_outputs = prev_outputs2,
                                                                                                                                                        phase_prev_outputs = prev_outputs1,
                                                                                                                                                        window_size = random_window_size,
                                                                                                                                                        mag_gates_remember = mag_gates_remember,
                                                                                                                                                        phase_gates_remember = phase_gates_remember,
                                                                                                                                                        eval = True
                                                                                                                                                    )

                prev_outputs2 = [(x - 5)*cycle_mask for x in prev_outputs2]
                prev_outputs1 = [(x - 4)*cycle_mask for x in prev_outputs1]
     
                
                phase_ti = torch.complex(torch.cos(prev_outputs1[-1]), torch.sin(prev_outputs1[-1]))

                lstm_predicted_fft = (self.param_dic['logarithm_base']**prev_outputs2[-1])*phase_ti
                predr_ti = torch.fft.ifft2(torch.fft.ifftshift(lstm_predicted_fft, dim = (-2,-1)))
                predr_ti = predr_ti.abs().clip(-10,10)
                
                if self.param_dic['image_lstm']:
                    B,C,numr, numc = predr_ti.shape
                    predr_ti = predr_ti.reshape(B*C, 1, numr, numc)
                    if prev_output3 is not None:
                        prev_output3 = prev_output3.reshape(B*C, 1, numr, numc)
                    prev_state3, prev_output3 = self.ispacem(predr_ti, prev_state3, prev_output3)
                    prev_output3 = prev_output3.reshape(B, C, numr, numc)
                    predr_ti = predr_ti.reshape(B, C, numr, numc)
                else:
                    prev_output3 = predr_ti
                
                predr[:,ti,:,:,:] = ispace_model(prev_output3).cpu()
                
                times.append(time.time() - start1)


        return times

    # AERS: note that the argument fft is fft_exp, meaning the exponential of. 
    def forward(self, fft_exp, gt_masks = None, device = torch.device('cpu'), periods = None, targ_phase = None, targ_mag_log = None, targ_real = None, og_video = None, epoch = np.inf):
        # AERS: Per Niraj's video from June 10th 2024, Part 2, min 15: Summary of what this function does:
        #  Forward pass where, given the sequence of video frames, we pass each frame to the network, one by one.
        # The LSTM has its own cell state and output. So that the LSTM returns a current state and output. These get appended to the LSTM, for the next loop iteration.
        # They are stored and appended again to the next forward pass. This is why it is called a recurrent network. 
        # For every for loop iteration, we get a prediction and have the ground truth. Thus, the loss is computed for every forward pass, and it is stored in a variable.
        # The predictions and losses are what is returned by this function.   

        # AERS: This line ensures that the log is not 0 or NaN (because where the mask is zeros, the log would be NaN). However, where there is no data, it should be 0.
        # What this does is that it clips the log so that all small values are -5 and adds 5 to it, so all the small values are 0. And the max value doesn't need to be limited.  
        mag_log = mylog(fft_exp.abs().clip(1e-5,1e20), base = self.param_dic['logarithm_base']) + 5

        # AERS: Computes the phase:
        phase = fft_exp / (EPS + fft_exp.abs())                 
        phase = torch.stack((phase.real, phase.imag), -1)       # AERS: cos(theta) + i*sin(theta). These are not combined into theta here because of testing of different conditions 
                                                                # like cosine mode, unit vector mode, etc. They get combined after that (~line 990: using torch.atan2)
 
        prev_outputs1 = None
        prev_outputs2 = None
        prev_state3 = None
        prev_output3 = None
        mag_gates_remember = None
        phase_gates_remember = None

        ans_coils = self.param_dic['num_coils']

        length = mag_log.shape[-1]
        x_size, y_size = length, length
        x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
        cell = (length//2, length//2)
        dists = ((x_arr - cell[0])**2 + (y_arr - cell[1])**2)**0.22
        dists = torch.FloatTensor(dists+1).to(device).unsqueeze(0).unsqueeze(0)
        if not self.param_dic['center_weighted_loss']:
            dists = torch.ones_like(dists, device = device).unsqueeze(0).unsqueeze(0)

        # AERS: Cycle mask are variables that will store the answers: mag_log, phase and predr.
        cycle_mask = ((x_arr - cell[0])**2 + (y_arr - cell[1])**2)**2
        cycle_mask[cycle_mask > cycle_mask[0,x_size//2]] = -1
        cycle_mask[cycle_mask > -1] = 1
        cycle_mask[cycle_mask == -1] = 0
        cycle_mask = torch.FloatTensor(cycle_mask).to(device).unsqueeze(0).unsqueeze(0)
        predr = torch.zeros(*mag_log.shape[0:2], ans_coils, *mag_log.shape[3:]).to(device)  # AERS: answers are the same size as the mag_log, but with multiple coils.

        # AERS: Declaring 0 variables. 
        if targ_mag_log is not None:                             # AERS: If targer phase is None (which means we are evaluating), then we do not send the ground truth and do not return all losses. This ensures that the ground truth is not used in evaluation/testing. 
            targ_mag_log *= cycle_mask.unsqueeze(0)
            targ_phase *= cycle_mask.unsqueeze(0).unsqueeze(-1)

        if self.param_dic['image_lstm']:
            predr_kspace = torch.zeros(*mag_log.shape[0:2], ans_coils, *mag_log.shape[3:])
        else:
            predr_kspace = None

        if targ_phase is not None:
            loss_phase = 0
            loss_mag = 0 
            loss_forget_gate = 0
            loss_input_gate = 0
            loss_real = 0
            loss_l1 = 0
            loss_l2 = 0
            loss_ss1 = 0
            criterionL1 = nn.L1Loss().to(device)
            criterionL2 = nn.MSELoss().to(device)
            criterionCos = nn.CosineSimilarity(dim = 4) 
        else:
            loss_phase = None
            loss_mag = None
            loss_real = None
            loss_forget_gate = None
            loss_input_gate = 0
            loss_l1 = 0
            loss_l2 = 0
            loss_ss1 = 0

        for ti in range(mag_log.shape[1]):          # AERS: Shape of mag_log is batch size (usually 1, B), number of frames (Nf), number of coils (Nc), Nr, Nr (resolution or matrix size)
            if periods is None:                     
                hist_phase = phase[:,ti]
                hist_mag = mag_log[:,ti]
            else:                                   # AERS: This is for ARKS (hist_mag, hist_phase, hist_ind). Actual ARKS angle need to be implemented. This is performing GA.
                # AERS: From June 10th, part 2, min 27 video
                # If Batch size = 3
                # periods = 30, 35 and 90 (originally 40) frames
                # history length = 2
                # number of frames that we have in mag_log = 120
                # and we are trying to predict ti (follow up index, same as frame?) 118
                
                # Solution:  (AERS: Need to double check, I didn't get to this exact solution because numbers get changed twice in the video.)
                #   1st video should get frames 118, 88, 58
                #   2nd video should get frames 118, 83, 48
                #   3rd video should get frames 118, 28, NaN (previously 118, 78, 38)
                
                hist_ind = (torch.arange(self.history_length+1).repeat(mag_log.shape[0],1) - self.history_length)
                # torch.arange(self.history_length+1) = [0, 1, 2], and repeated mag_log.shape[0] (which is the number of videos in the batch). So:
                # hist_ind [0, 1, 2] - self.history_length = [-2, -1, 0] for all 3 videos, so:
                # hist_ind = [-2, -1, 0]
                #            [-2, -1, 0]
                #            [-2, -1, 0]

                hist_ind = hist_ind * periods.reshape(-1,1).cpu()
                # periods = [30, 35, 40] becomes a column and multplied times hist_ind results in:
                # hist_ind = [-60, -30, 0]
                #            [-70, -35, 0]
                #            [-180, -90, 0] (for a period of 90, rather than 40)

                hist_ind += ti
                # hist_ind = [118-60, 118-30, 118]
                #            [118-70, 118-35, 118]
                #            [118-180, 118-90, 118]

                temp1 = hist_ind.clone()                # AERS: Then it gets cloned into temp1
                # temp1 = [118-60, 118-30, 118]
                #         [118-70, 118-35, 118]
                #         [118-180, 118-90, 118]
                                
                # AERS: For example, if we are at frame 38, and have a history of 3. There is only one previous history timepoint. And the others do not exist. Those values of temp1 become negative and replaced with 9999999999.
                temp1[temp1 < 0] = 9999999999
                # temp1 = [118-60, 118-30, 118]
                #         [118-70, 118-35, 118]
                #         [9999999999, 118-90, 118]     # AERS: Assuming only this one doesn't exist

                min_vals = temp1.min(1)[0]              # AERS: Returns the min values along the columns direction. [0] specifies the values, [1] return the indexes
                # min_vals = [118-60]
                #            [118-70]
                #            [118-90]  
                
                base = torch.zeros(temp1.shape) + min_vals.reshape(-1,1)  
                # base = [118-60, 118-60, 118-60]
                #        [118-70, 118-70, 118-70]
                #        [118-90, 118-90, 118-90]

                hist_ind = hist_ind - base
                # hist_ind = [118-60, 118-30, 118] - [118-60, 118-60, 118-60]
                #            [118-70, 118-35, 118] - [118-70, 118-70, 118-70]
                #            [118-90, 118-90, 118] - [118-90, 118-90, 118-90]

                # hist_ind = [0, 30, 60]
                #            [0, 35, 70]
                #            [-90, 0, 90 ]

                hist_ind[hist_ind < 0] = 0
                # hist_ind = [0, 30, 60]
                #            [0, 35, 70]
                #            [0, 0, 90 ]

                hist_ind = (hist_ind + base).long() 
                # hist_ind = [118-60, 118-30, 118]
                #            [118-70, 118-35, 118]
                #            [118-90, 118-90, 118]

                mult = (torch.arange(mag_log.shape[0])*mag_log.shape[1]).reshape(-1,1)
                # mult = [ 0 ]
                #        [120]
                #        [240]

                hist_ind = hist_ind + mult      
                # hist_ind = [  118-60+0,   118-30+0,   118+0] = [ 58,  88, 118]
                #            [118-70+120, 118-35+120, 118+120] = [168, 203, 238]
                #            [118-90+240, 118-90+240, 118+240] = [268, 268, 358]                

                hist_phase = phase.reshape(-1, *phase.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *phase.shape[3:])
                 # phase then becomes:
                 # phase = B*Nf, Nc, Nr, Nr , instead of B, Nf, Nc, Nr, Nr          # AERS: This way, the first 120 frames, will be the first batch, the second 120 frames, will be the 2nd batch, and so on.

                hist_mag = mag_log.reshape(-1, *mag_log.shape[2:])[hist_ind.reshape(-1)].reshape(hist_ind.shape[0], -1, *mag_log.shape[3:])

            background = ((hist_phase[:,:,:,:,1].abs() + hist_phase[:,:,:,:,0].abs()) < 1)
            # AERS: Take the atan of first and second coordinate, which are sin and cos. Results in the phase angles, but the background needs to be 0s.
            #       The tange of the output of ata2 is -pi to pi. However, with the background = 0, it would be confusing for the NN (it would think that 0 is a valid angle). 
            #       So, an offset of 4 is added to everything and then multiplied times the background mask. The important part is that the phase now ranges from 4-pi to 4+pi, not passing by 0.
            hist_phase = torch.atan2(hist_phase[:,:,:,:,1]+EPS,hist_phase[:,:,:,:,0]) + 4       
            hist_phase[background] = 0
            hist_mag[background] = 0

            if prev_outputs2 is not None:
                prev_outputs2 = [(x + 5) for x in prev_outputs2]
                prev_outputs1 = [x + 4 for x in prev_outputs1]
            if gt_masks is None:
                curr_mask = None
            else:
                curr_mask = gt_masks[:,ti,:,:,:]
            if epoch < self.param_dic['num_epochs_recurrent'] - self.param_dic['num_epochs_windowed']:
                random_window_size = np.inf
            else:
                # AERS: Forward pass for the k-space RNN. A single cell of a k-space RNN–every cell is going to have multiple forward passes for every frame.
                random_window_size = np.random.choice(self.param_dic['window_size'])
            
            # AERS: This is the actual forward pass!
            # AERS: Steady-state loop invariant: prev_outputs2 (gt magnitude) and prev_outputs1 (gt phase). Then 4 and 5 are subtracted to get them to their original value.
            prev_outputs2, prev_outputs1, loss_forget_gate_curr, loss_input_gate_curr, mag_gates_remember, phase_gates_remember = self.kspace_m(
                                                                                                                                                        hist_mag,
                                                                                                                                                        hist_phase,
                                                                                                                                                        background = background,
                                                                                                                                                        gt_mask = curr_mask,
                                                                                                                                                        mag_prev_outputs = prev_outputs2,
                                                                                                                                                        phase_prev_outputs = prev_outputs1,
                                                                                                                                                        window_size = random_window_size,
                                                                                                                                                        mag_gates_remember = mag_gates_remember,
                                                                                                                                                        phase_gates_remember = phase_gates_remember,
                                                                                                                                                        eval = False
                                                                                                                                                    )
            # AERS: about the mag_gates_remember and phase_gates_remember: These are the forget and input gates, they are the same thing because forget_gate(t) = input_gate(t-1).
            #       In this code, the inpute gates or forget gates of all previous gates are stacked. However, if the search window is, for example, 2 frames. 
            #       Then, the code checks what is remembered for the last 2 frames and it is the only thing that is kept from the previous cell state. Everything else from previous cell state is deleted.
            #       The window size is controlled (I believe in the parameters), but actually, it doesn't have to be a fixed value. It could be varied for example, if we check the ECG and we are too far 
            #       from a period of interest (systole or diastole), then we can increase the window size. As sytole/diastole approaches, we can then decrease the window size and reduce the lag.

            # AERS: the forget get mask will suppose there is new data added at a location and will forget the previous state or data of that location regardless of the previous value.
            # However, this will only work for the spokes, not for background/edges? regions. When the FT is computer to try and predict the image space in a future frame, there will be -5 or -4 values, giving incorrect results.
            # So, we need to multiply by a mask that zeros out these values.

            # AERS: These are the predictions for the k-space RNN
            prev_outputs2 = [(x - 5)*cycle_mask for x in prev_outputs2]     # Magnitude
            prev_outputs1 = [(x - 4)*cycle_mask for x in prev_outputs1]     # Phase

            del hist_mag
            del hist_phase
            
            # AERS: for loop over cells (comments assume only one cell exists)
            for i_cell in range(self.param_dic['n_rnn_cells']):

                # AERS: The mode was set as theta, so the phase is computed as cos,sine and we have a stacked phase. Then the FT is computed.
                phase_ti = torch.complex(torch.cos(prev_outputs1[i_cell]), torch.sin(prev_outputs1[i_cell]))
                stacked_phase = torch.stack((phase_ti.real, phase_ti.imag), -1)

                # AERS: This is the FFT predicted by the k-space RNN, not LSTM (variable name was not updated from previous versions). Then take the inverse FT (predr_ti) and clip values
                lstm_predicted_fft = (self.param_dic['logarithm_base']**prev_outputs2[i_cell])*phase_ti
                predr_ti = torch.fft.ifft2(torch.fft.ifftshift(lstm_predicted_fft, dim = (-2,-1)))          
                predr_ti = predr_ti.abs().clip(-10,10)                                                         # AERS: Image-based prediction of the k-space RNN

                if self.param_dic['image_lstm']:   # AERS: If image-space LSTM is active (according to the parameters). prev_output3 is the final prediction predr that is returned
                    B,C,numr, numc = predr_ti.shape
                    predr_ti = predr_ti.reshape(B*C, 1, numr, numc)
                    if prev_output3 is not None:
                        prev_output3 = prev_output3.reshape(B*C, 1, numr, numc)
                    if self.param_dic['unet_instead_of_ilstm']:
                        prev_output3 = self.ispacem(predr_ti)
                    else:
                        prev_state3, prev_output3 = self.ispacem(predr_ti, prev_state3, prev_output3)
                    prev_output3 = prev_output3.reshape(B, C, numr, numc)
                    predr_ti = predr_ti.reshape(B, C, numr, numc)
                else:
                    prev_output3 = predr_ti.clone()
                
                
                ########## LOSS FUNCTION ##########
                # AERS: For the first few frames, the results are really bad because the memory has not accumulated yet. So the loss are not computed for those (init_skip_frames). 
                if ti >= self.param_dic['init_skip_frames']:
                    if targ_phase is not None:
                        
                        # AERS: (?) This is the loos for the k-space RNN. Crop loss:
                        loss_mag += criterionL1((dists*prev_outputs2[i_cell]), dists*cycle_mask*(targ_mag_log[:,ti,:,:,:].to(device)))/(mag_log.shape[1]*self.param_dic['n_rnn_cells'])

                        # AERS: Forget gate and input gate loss. These are returned by the k-sapce RNN forward pass. This just adds those losses.
                        loss_forget_gate += loss_forget_gate_curr/(mag_log.shape[1]*self.param_dic['n_rnn_cells'])
                        loss_input_gate += loss_input_gate_curr/(mag_log.shape[1]*self.param_dic['n_rnn_cells'])

                        # AERS: To compute the target angles atan2 of the gt and the phase loss is computed as the L1 loss of the previous output (k-space RNN phase predictions) and the target angles.
                        #       This is the loss for the k-space magnitude, phase and the gates 
                        targ_angles = torch.atan2((targ_phase[:,ti,:,:,:,1])+EPS,(targ_phase[:,ti,:,:,:,0])).to(device)
                        loss_phase += criterionL1(prev_outputs1[i_cell], cycle_mask*targ_angles)/(mag_log.shape[1]*self.param_dic['n_rnn_cells'])
                        
                        # AERS: We want to have a loss on the real data, as well. So here, it is calculated.
                        targ_now = targ_real[:,ti,:,:,:].to(device)

                        if prev_output3 is not None: # AERS: If there was an image LSTM, then prev_output3 will have some values and it will calculate the L2 loss on those data
                            if epoch < self.param_dic['num_epochs_recurrent'] - self.param_dic['num_epochs_ilstm']:
                                loss_real += 1e-10*criterionL2(prev_output3*self.lossmask, targ_now*self.lossmask)/(mag_log.shape[1]*self.param_dic['n_rnn_cells'])
                            else:
                                loss_real += 8*criterionL2(prev_output3*self.lossmask, targ_now*self.lossmask)/(mag_log.shape[1]*self.param_dic['n_rnn_cells'])
                        loss_real += criterionL2(predr_ti*self.lossmask, targ_now*self.lossmask)/(mag_log.shape[1]*self.param_dic['n_rnn_cells'])
                
                        # AERS: other output loss statistics 
                        with torch.no_grad():
                            loss_l1 += (predr_ti- targ_now).reshape(predr_ti.shape[0]*predr_ti.shape[1], -1).abs().mean(1).sum().detach().cpu()/self.param_dic['n_rnn_cells']
                            loss_l2 += (((predr_ti- targ_now).reshape(predr_ti.shape[0]*predr_ti.shape[1], -1) ** 2).mean(1).sum()).detach().cpu()/self.param_dic['n_rnn_cells']
                            ss1 = self.SSIM(predr_ti.reshape(predr_ti.shape[0]*predr_ti.shape[1],1,self.param_dic['image_resolution'],self.param_dic['image_resolution']), targ_now.reshape(predr_ti.shape[0]*predr_ti.shape[1],1,self.param_dic['image_resolution'],self.param_dic['image_resolution']))
                            ss1 = ss1.reshape(ss1.shape[0],-1)
                            loss_ss1 += ss1.mean(1).sum().detach().cpu() / (self.param_dic['n_rnn_cells'])

            # AERS: For the image LSTM, the output is detached (here prev_output3), but not the cell state. 
            # Meaning, detach from the tree, so that the tree can pursue the tree path when the final loss backwards (video Juen 10th Part 3, min 44)
            if self.param_dic['image_lstm']:
                prev_output3 = prev_output3.detach()
            
            # AERS: previous outputs are detached after each for loop, and same for the image LSTM
            if self.param_dic['image_lstm']:
                predr_kspace[:,ti,:,:] = predr_ti.detach().cpu()
            predr[:,ti,:,:] = prev_output3.detach()

        predr = predr * self.predr_mask

        return predr, predr_kspace, loss_mag, loss_phase, loss_real, loss_forget_gate, loss_input_gate, (loss_l1, loss_l2, loss_ss1)


class CoupledDown(nn.Module):
    def __init__(self, inp_channel, outp_channels):
        super(CoupledDown, self).__init__()
        ls = []
        prev = inp_channel
        for i in outp_channels:
            ls.append(cmplx_conv.ComplexConv2d(prev, i, (3,3), stride = (1,1), padding = (1,1), bias = False))
            ls.append(cmplx_activation.CReLU())
            ls.append(radial_bn.RadialBatchNorm2d(i))
            prev = i
        self.model = nn.Sequential(*ls)
        self.final = cmplx_conv.ComplexConv2d(prev, prev, (2,2), stride = (2,2), padding = (0,0))
        self.train_mode = True

    def forward(self, x):
        if self.train_mode:
            x1 = self.model(x)
        else:
            x1 = no_bn_forward(self.model, x)
        return x1, self.final(x1)

    def train_mode_set(self, bool = True):
        self.train_mode = bool

class CoupledUp(nn.Module):
    def __init__(self, inp_channel, outp_channels):
        super(CoupledUp, self).__init__()
        ls = []
        prev = inp_channel
        for i in outp_channels:
            ls.append(cmplx_conv.ComplexConv2d(prev, i, (3,3), stride = (1,1), padding = (1,1), bias = False))
            ls.append(cmplx_activation.CReLU())
            ls.append(radial_bn.RadialBatchNorm2d(i))
            prev = i
        self.model = nn.Sequential(*ls)
        self.final = cmplx_upsample.ComplexUpsample(scale_factor = 2, mode = 'bilinear')
        self.train_mode = True

    def forward(self, x):
        if self.train_mode:
            x1 = self.model(x)
        else:
            x1 = no_bn_forward(self.model, x)
        return self.final(x1)

    def train_mode_set(self, bool = True):
        self.train_mode = bool

class CoupledDownReal(nn.Module):
    def __init__(self, inp_channel, outp_channels):
        super(CoupledDownReal, self).__init__()
        ls = []
        prev = inp_channel
        for i in outp_channels:
            ls.append(nn.Conv2d(prev, i, (3,3), stride = (1,1), padding = (1,1), bias = False))
            ls.append(nn.ReLU())
            ls.append(nn.BatchNorm2d(i))
            prev = i
        self.model = nn.Sequential(*ls)
        self.final = nn.Conv2d(prev, prev, (2,2), stride = (2,2), padding = (0,0))
        # self.final = nn.Sequential(
        #                                     nn.Conv2d(prev, prev, (2,2), stride = (2,2), padding = (0,0)),
        #                                     nn.ReLU(),
        #                                     nn.BatchNorm2d(prev)
        #                             )
        self.train_mode = True

    def forward(self, x):
        if self.train_mode:
            x1 = self.model(x)
        else:
            x1 = no_bn_forward(self.model, x)
        return x1, self.final(x1)

    def train_mode_set(self, bool = True):
        self.train_mode = bool

class CoupledUpReal(nn.Module):
    def __init__(self, inp_channel, outp_channels):
        super(CoupledUpReal, self).__init__()
        ls = []
        prev = inp_channel
        for i in outp_channels:
            ls.append(nn.Conv2d(prev, i, (3,3), stride = (1,1), padding = (1,1), bias = False))
            ls.append(nn.ReLU())
            ls.append(nn.BatchNorm2d(i))
            prev = i
        self.model = nn.Sequential(*ls)
        self.final = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        self.train_mode = True

    def forward(self, x):
        if self.train_mode:
            x1 = self.model(x)
        else:
            x1 = no_bn_forward(self.model, x)
        return self.final(x1)

    def train_mode_set(self, bool = True):
        self.train_mode = bool

# AERS: MDCNN architecture
class ImageSpaceModel1(nn.Module): 
    def __init__(self, parameters, proc_device):
        super(ImageSpaceModel1, self).__init__()
        self.param_dic = parameters
        self.final_prediction_real = self.param_dic['final_prediction_real']
        self.num_coils = self.param_dic['num_coils']
        if self.param_dic['coil_combine'] == 'SOS':
            self.input_size = 1
        else:
            self.input_size = self.param_dic['num_coils']
        if self.final_prediction_real:

            # AERS: Block 1 (Described in Video June 10th, Part 3, 46:30 min)
            self.block1 = nn.Sequential(
                    nn.Conv2d(self.input_size, 16, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm2d(16),
                    nn.Conv2d(16, self.input_size, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm2d(self.input_size),
                )
            self.down1 = CoupledDownReal(self.input_size, [32,32])
            self.down2 = CoupledDownReal(32, [64,64])
            self.down3 = CoupledDownReal(64, [128,128])
            self.up1 = CoupledUpReal(128, [256,128])
            self.up2 = CoupledUpReal(256, [128,64])
            self.up3 = CoupledUpReal(128, [64,32])
            self.finalblock = nn.Sequential(
                    nn.Conv2d(64, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    nn.ReLU(),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 1, (3,3), stride = (1,1), padding = (1,1)),
                )
        else:
            self.block1 = nn.Sequential(
                    cmplx_conv.ComplexConvd(self.num_coils, 2*self.num_coils, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNormd(2*self.num_coils),
                    cmplx_conv.ComplexConvd(2*self.num_coils, self.num_coils, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNormd(self.num_coils)
                )
            self.down1 = CoupledDown(self.num_coils, [32,32])
            self.down2 = CoupledDown(32, [64,64])
            self.down3 = CoupledDown(64, [128,128])
            self.up1 = CoupledUp(128, [256,128])
            self.up2 = CoupledUp(256, [128,64])
            self.up3 = CoupledUp(128, [64,32])
            self.finalblock = nn.Sequential(
                    cmplx_conv.ComplexConv2d(64, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNorm2d(32),
                    cmplx_conv.ComplexConv2d(32, 32, (3,3), stride = (1,1), padding = (1,1), bias = False),
                    cmplx_activation.CReLU(),
                    radial_bn.RadialBatchNorm2d(32),
                    cmplx_conv.ComplexConv2d(32, 1,     (3,3), stride = (1,1), padding = (1,1)),
                )

    def time_analysis(self, x, device):
        start = time.time()
        with torch.no_grad():
            x1 = self.block1(x)
            x2hat, x2 = self.down1(x1)
            x3hat, x3 = self.down2(x2)
            x4hat, x4 = self.down3(x3)
            x5 = self.up1(x4)
            x6 = self.up2(torch.cat((x5,x4hat),1))
            x7 = self.up3(torch.cat((x6,x3hat),1))
            x8 = self.finalblock(torch.cat((x7,x2hat),1))
        return time.time() - start

    # AERS: These is where the connections of the MCDNN are defined
    def forward(self, x):
        x1 = self.block1(x)+x
        x2hat, x2 = self.down1(x1)
        x3hat, x3 = self.down2(x2)
        x4hat, x4 = self.down3(x3)
        x5 = self.up1(x4)
        x6 = self.up2(torch.cat((x5,x4hat),1))
        x7 = self.up3(torch.cat((x6,x3hat),1))
        x8 = self.finalblock(torch.cat((x7,x2hat),1))
        return x8





class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool2(enc2))

        # Decoder
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)