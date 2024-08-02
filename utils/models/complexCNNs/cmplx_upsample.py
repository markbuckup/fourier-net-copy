import torch
import torch.nn as nn


class ComplexUpsample(nn.Module):
    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode='nearest',
        align_corners=False,
        recompute_scale_factor=False,
    ):
        """
        Upsample layer for complex inputs.

        Parameters:
        ------------
        - size : int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional
            Output spatial sizes
        - scale_factor : float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional
            Multiplier for spatial size. Has to match input size if it is a tuple.
        - mode : str, optional
            The upsampling algorithm: one of ``'nearest'``, ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
                Default: ``'nearest'``
        - align_corners : bool, optional
            If ``True``, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels. \
                This only has effect when :attr:`mode` is ``'linear'``, ``'bilinear'``, ``'bicubic'``, or ``'trilinear'``.
                Default: ``False``
        - recompute_scale_factor : bool, optional
            Recompute the scale_factor for use in the interpolation calculation. If `recompute_scale_factor` is ``True``, then \
            `scale_factor` must be passed in and `scale_factor` is used to compute the output `size`. The computed output `size`\
            will be used to infer new scales for the interpolation. 
            
            **Note:** When `scale_factor` is floating-point, it may differ from the recomputed `scale_factor` due to rounding \
                and precision issues. If `recompute_scale_factor` is ``False``, then `size` or `scale_factor` will be used directly for interpolation.

        ================================================================================================
        """
        super(ComplexUpsample, self).__init__()
        self.upsample = nn.Upsample(
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor
        )

    def forward(self, input):
        """
        AERS:
        Forward pass for the upsampling of complex-valued inputs.

        This method takes a complex-valued input tensor, separates it into its real and imaginary components,
        applies an upsampling operation to both components independently, and then recombines them into a 
        complex-valued output tensor.

        Parameters:
        ------------
        - input : torch.Tensor
            A complex-valued tensor with shape (..., 2), where the last dimension contains 
            the real and imaginary parts of the input.

        Returns:
        ---------
        - torch.Tensor
            A complex-valued tensor with the same shape as the input, where each part has been upsampled.

        ================================================================================================
        """
        real, imag = torch.unbind(input, dim=-1)
        output = torch.stack((self.upsample(real), self.upsample(imag)), dim=-1)
        return output
