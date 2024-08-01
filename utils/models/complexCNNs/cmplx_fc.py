import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    """
    AERS A linear transformation layer for complex-valued inputs.

    This module applies separate linear transformations to the real and imaginary parts of the input tensor.
    """
    def __init__(self, in_features, out_features, bias=True):
        """
        AERS Initializes the ComplexLinear layer.

        Args:
        --------
        in_features : int
            The number of input features.
        out_features : int
            The number of output features.
        bias : bool, optional
            If True, includes a bias term in the linear transformations. 
                Default is True.
        """
        super(ComplexLinear, self).__init__()
        self.real_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.imag_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, input):
        """
        AERS Forward pass of the ComplexLinear layer.

        This method splits the input tensor into its real and imaginary components, applies the linear transformations
        to each component, and then recombines them to form the complex-valued output.

        Args:
        --------
        input : torch.Tensor
            A complex-valued tensor with shape (..., 2), where the last dimension contains the real and imaginary parts of the input.

        Returns:
        --------
        torch.Tensor
            A complex-valued tensor with the same shape as the input, where each part has been linearly transformed.
        """
        real, imag = torch.unbind(input, dim=-1)

        real_out = self.real_linear(real) - self.imag_linear(imag)
        imag_out = self.real_linear(imag) + self.imag_linear(real)

        output = torch.stack((real_out, imag_out), dim=-1)

        return output
