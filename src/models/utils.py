"""
Blocks and layers used by both generator and discriminator.
"""

from torch import nn
from src.settings import STD


class NormalConv2d(nn.Conv2d):
    """
    A 2d convolutional layer that is initialized using a normal distribution.

    The standard deviation of the normal distribution is controled by the constant
    `STD`. The bias of the layer is initialized to zero.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args: iterable
            Arguments for the parent class.
        **kwargs: dict
            Keyword arguments for the parent class.
        """
        super(NormalConv2d, self).__init__(*args, **kwargs)
        self.weight.data.normal_(std=STD)
        self.bias.data.zero_()


class NormalLinear(nn.Linear):
    """
    A linear layer that is initialized using a normal distribution.

    The standard deviation of the normal distribution is controled by the constant
    `STD`. The bias of the layer is initialized to zero.
    """
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        *args: iterable
            Arguments for the parent class
        **kwargs: dict
            Keyword arguments for the parent class
        """
        super(NormalLinear, self).__init__(*args, **kwargs)
        self.weight.data.normal_(std=STD)
        self.bias.data.zero_()


class FromRGB(nn.Module):
    """
    Block used to transform a RGB tensor image.

    From having 3 channels the tensor image goes to have `out_channels`.

    Attributes
    ----------
    conv: NormalConv2d
        2d convolutional layer that receives a tensor with 3 channels and returns a tensor
        with `out_channels`.
    act: LeakyReLU
        Activation.

    Methods
    -------
    forward(x)
        Receives an image tensor with 3 channels and returns a tensor with `out_channels`.
    """
    def __init__(self, out_channels):
        """
        Parameters
        ----------
        out_channels: int
            Number of channels of the output.
        """
        super(FromRGB, self).__init__()
        self.conv = NormalConv2d(3, out_channels, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Receives an image tensor with 3 channels and returns a tensor with `out_channels`.

        Parameters
        ----------
        x: tensor
            Tensor image with 3 channels (RGB).

        Returns
        -------
        tensor
            A tensor image with `out_channels`.
        """
        x = self.conv(x)
        return self.act(x)


class ToRGB(nn.Module):
    """
    Wrapper class to a convolutional layer that takes an input tensor
    with `input_channels` and outputs a tensor with 3 channels.

    Attributes
    ----------
    conv: NormalConv2d
        2d convolution used to change the number of channels of the input.

    Methods
    -------
    forward(x)
        Outputs a new tensor with 3 channels (RGB).
    """
    def __init__(self, input_channels, *args, **kwargs):
        """
        Parameters
        ----------
        input_channels: int
            Number of channels of the input tensor
        """
        super(ToRGB, self).__init__(*args, **kwargs)
        self.conv = NormalConv2d(input_channels, 3, 1)

    def forward(self, x):
        """
        Outputs a tensor with 3 channels (RGB) after applying a convolution to the input
        tensor `x`.

        Parameters
        ----------
        x: tensor
            The image tensor to be convoluted.

        Returns
        -------
        tensor
            The new tensor with 3 channels (RGB) resulting of convoluting the input.
        """
        return self.conv(x)
