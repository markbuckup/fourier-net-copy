import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from utils.models.complexCNNs.polar_transforms import (
    convert_polar_to_cylindrical,
    convert_cylindrical_to_polar
)


class CReLU(nn.ReLU):
    def __init__(self, inplace: bool=False):
        """
        AERS:
        Custom ReLU activation function that extends the PyTorch nn.ReLU class.

        This class implements a variation of the Rectified Linear Unit (ReLU) activation function.
        
        Paramaters:
        ------------
        - inplace : bool, optional 
            If set to True, will do the operation in-place without using extra memory for a new tensor.
                Default=False
        
        ================================================================================================
        """
        super(CReLU, self).__init__(inplace)
    

class ModReLU(nn.Module):
    def __init__(self, in_channels, inplace=True):
        """ModReLU

        Paramaters:
        ------------
        - in_channels : int
            The number of input channels.
        - inplace : bool
            If True, the input is modified.

        ================================================================================================
        """
        super(ModReLU, self).__init__()
        self.inplace = inplace
        self.in_channels = in_channels
        self.b = Parameter(torch.Tensor(in_channels), requires_grad=True)
        self.reset_parameters()
        self.relu = nn.ReLU(self.inplace)

    def reset_parameters(self):
        """
        AERS:
        Resets the parameters of the module to their initial state.

        This method initializes the parameter `b` of the module by setting its values to a uniform distribution
        within the range [-0.1, 0.1].

        Returns:
        --------
        None.

        ================================================================================================
        """
        self.b.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        """
        AERS 
        Forward pass of the ModReLU layer.

        This method takes a complex-valued input tensor, separates it into its real and imaginary components, 
        converts these components from cylindrical to polar coordinates, applies the ModReLU activation 
        function on the magnitude, and then converts the result back to cylindrical coordinates. 
        The output is a complex-valued tensor.

        Paramaters:
        -----------
        - input : torch.Tensor
            A complex-valued tensor with shape `(..., 2)`, where the last dimension contains the real and imaginary parts of the input.

        Returns:
        --------
        - output : torch.Tensor
            A complex-valued tensor with the same shape as the input, where the ModReLU activation function has been applied.

        Notes:
        ------
        - The `input` tensor is assumed to be in a format where the last dimension contains the real and imaginary parts (i.e., `[..., 2]`).
        - The method relies on helper functions `convert_cylindrical_to_polar` and `convert_polar_to_cylindrical` for coordinate transformations.
        - The activation function applied is the ReLU function with an added bias term broadcasted to match the shape of the magnitude component.

        ================================================================================================
        """    
        real, imag = torch.unbind(input, -1)
        mag, phase = convert_cylindrical_to_polar(real, imag)
        brdcst_b = torch.swapaxes(torch.broadcast_to(self.b, mag.shape), -1, 1)
        mag = self.relu(mag + brdcst_b)
        real, imag = convert_polar_to_cylindrical(mag, phase)
        output = torch.stack((real, imag), dim=-1)
        return output


class ZReLU(nn.Module):
    """
    AERS:
    ZReLU activation function.

    This class implements the ZReLU activation function, which applies a ReLU-like operation on complex-valued inputs, \
        zeroing out parts of the input based on the phase of the complex numbers.

    ================================================================================================
    """
    def __init__(self):
        """
        AERS:
        Initializes the ZReLU activation function.

        This is a basic initialization function that calls the parent class's initializer.

        ================================================================================================
        """
        super(ZReLU, self).__init__()

    def forward(self, input):
        """
        AERS: 
        Forward pass of the ZReLU layer.

        This method processes a complex-valued input tensor by separating it into its real 
        and imaginary components, converting them to polar coordinates, and then applying 
        a condition where only complex numbers with a phase between 0 and π/2 are retained, 
        while others are set to zero.

        Paramaters:
        -----------
        - input : torch.Tensor
            A complex-valued tensor with shape (..., 2), where the last dimension contains 
            the real and imaginary parts of the input.

        Returns:
        --------
        - output : torch.Tensor
            A complex-valued tensor with the same shape as the input, where elements that 
            do not meet the phase condition are set to zero.

        ================================================================================================
        """
        real, imag = torch.unbind(input, dim=-1)
        mag, phase = convert_cylindrical_to_polar(real, imag)

        phase = torch.stack([phase, phase], dim=-1)
        output = torch.where(phase >= 0.0, input, torch.tensor(0.0).to(input.device))
        output = torch.where(phase <= np.pi / 2, output, torch.tensor(0.0).to(input.device))

        return output
