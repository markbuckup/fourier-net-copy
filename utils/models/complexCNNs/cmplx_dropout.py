import torch
import torch.nn as nn


class ComplexDropout(nn.Module):

    def __init__(self, rank, p=0.5, inplace=True):
        """
        AERS:
        Initializes the ComplexDropout layer.

        This method sets up the ComplexDropout layer with the specified rank, dropout probability, 
        and whether the operation should be performed in-place.

        Parameters:
        -----------
        - rank : int
            The rank of the tensor to which the dropout will be applied.
        - p : float, optional
            Probability of dropping out elements in the tensor. Must be between 0 and 1. 
                Default is 0.5.
        - inplace : bool, optional
            If True, performs the operation in-place without using extra memory for a new tensor. 
                Default is True.

        Raises:
        -------
        - ValueError
            If the dropout probability `p` is not between 0 and 1.

        ==========================================================================
        """
        super(ComplexDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.rank = rank
        self.p = p
        self.inplace = inplace

    def extra_repr(self):
        """
        AERS:
        Returns a string representation of the layer's configuration.

        This method provides an extra representation of the `ComplexDropout` layer's settings, specifically the dropout probability (`p`) and whether the operation is performed in-place.

        Returns:
        --------
        -str
            A string that represents the dropout probability and the in-place setting of the layer. The string format is 'p={}' followed by ', inplace' if `inplace` is True.

        ==========================================================================
        """
        inplace_str = ', inplace' if self.inplace else ''
        return 'p={}{}'.format(self.p, inplace_str)

    def forward(self, input):
        """
        AERS:
        Applies dropout to the input during training.

        This method applies dropout to the input tensor by randomly setting a portion of the elements to zero 
        based on the dropout probability `p`. During evaluation (when `self.training` is `False`), 
        or if `p` is 0, the input is returned unchanged. If `p` is 1, the entire input is zeroed out.

        Parameters:
        -----------
        - input : torch.Tensor
            The input tensor to which dropout will be applied. The tensor is expected to be a complex-valued tensor with the last dimension representing real and imaginary components.

        Returns:
        --------
        - torch.Tensor
            The tensor after applying dropout. If the layer is in training mode, a portion of the elements will be zeroed based on the dropout probability `p`. If in evaluation mode or if `p` is 0, the input is returned unchanged.

        Notes:
        ------
        - The dropout mask is applied equally to both the real and imaginary parts of the input tensor.
        - The method handles edge cases where `p` is 0 or 1 to avoid unnecessary computations.

        ==========================================================================
        """    
        if not self.training or self.p == 0:
            return input

        if self.p == 1:
            return torch.FloatTensor(input.shape).to(input.device).zero_()

        msk = torch.FloatTensor(input.shape[:-1]).to(input.device).uniform_() > self.p
        msk = torch.stack([msk, msk], dim=-1)

        output = input * msk.to(torch.float32)

        return output


class ComplexDropout1d(ComplexDropout):
    r"""
    Randomly zeroes whole channels of the complex input tensor.
    The channels to zero are randomized on every forward call.
    Usually the input comes from :class:`nn.Conv3d` modules.

    Paramaters:
    -----------
    - p : float, optional
        Probability of an element to be zeroed.
    - inplace : bool, optional
        If set to ``True``, will do this operation in-place

    Shape:
    -----------
        - Input: :math:`(N, C, D, H, W, 2)`
        - Output: :math:`(N, C, D, H, W, 2)` (same shape as input)

    ================================================================================================
    """
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropout1d, self).__init__(
            rank=1,
            p=p,
            inplace=inplace
        )


class ComplexDropout2d(ComplexDropout):
    r"""
    Randomly zeroes whole channels of the complex input tensor.
    The channels to zero-out are randomized on every forward call.
    Usually the input comes from :class:`nn.Conv2d` modules.

    Paramaters:
    -----------
    - p : float, optional
        Probability of an element to be zero-ed.
    - inplace : bool, optional
        If set to ``True``, will do this operation in-place

    Shape:
    -----------
        - Input: :math:`(N, C, H, W, 2)`
        - Output: :math:`(N, C, H, W, 2)` (same shape as input)

    ================================================================================================
    """
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropout2d, self).__init__(
            rank=2,
            p=p,
            inplace=inplace
        )


class ComplexDropout3d(ComplexDropout):
    r"""
    Randomly zeroes whole channels of the complex input tensor.
    The channels to zero are randomized on every forward call.
    Usually the input comes from :class:`nn.Conv3d` modules.

    Paramaters:
    -----------
    - p : float, optional
        Probability of an element to be zeroed.
    - inplace : bool, optional 
        If set to ``True``, will do this operation in-place

    Shape:
    -----------
        - Input: :math:`(N, C, D, H, W, 2)`
        - Output: :math:`(N, C, D, H, W, 2)` (same shape as input)

    ================================================================================================
    """
    def __init__(self, p=0.5, inplace=False):
        super(ComplexDropout3d, self).__init__(
            rank=3,
            p=p,
            inplace=inplace
        )


if __name__ == '__main__':
    x = torch.rand((2, 2, 8, 8, 2))
    print(f"non-zero elements: {len(x.nonzero())}")
    dropout = ComplexDropout2d(p=0.5)
    y = dropout(x)
    print(f"non-zero elements: {len(y.nonzero())}")
