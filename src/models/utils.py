"""
Blocks and layers used by both generator and discriminator.
"""

from torch import nn
from settings import STD, DATASET_CHANNELS


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


class FromColorChannels(nn.Module):
    """
    Block used to transform the number of channels in a tensor image.

    From having the number of channels specified in the training dataset the tensor image goes to
    have an specified number of channels.

    Attributes
    ----------
    conv: NormalConv2d
        2d convolutional layer that receives a tensor with certain number of channels and returns
        a tensor with `out_channels`.
    act: LeakyReLU
        Activation.

    Methods
    -------
    forward(x)
        Receives an image tensor with certain number of channels and returns a tensor
        with `out_channels`.
    """
    def __init__(self, out_channels, color_channels=DATASET_CHANNELS):
        """
        Parameters
        ----------
        out_channels: int
            Number of channels of the output.
        color_channels: int, optional
            Number of channels in the images of the dataset. For example, the value of this variable
            for an RGB image would be 3, 1 for a black and white image.
            Default value is `DATASET_CHANNELS`.
        """
        super(FromColorChannels, self).__init__()
        self.conv = NormalConv2d(color_channels, out_channels, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Receives an image tensor with ertain number of channels and returns a tensor
        with `out_channels`.

        Parameters
        ----------
        x: tensor
            Tensor image.

        Returns
        -------
        tensor
            A tensor image with `out_channels`.
        """
        x = self.conv(x)
        return self.act(x)


class ToColorChannels(nn.Module):
    """
    Wrapper class to a convolutional layer that takes an input tensor and outputs a tensor
    with the specified number of channels.

    Attributes
    ----------
    conv: NormalConv2d
        2d convolution used to change the number of channels of the input.

    Methods
    -------
    forward(x)
        Outputs a new tensor with a specified number of channels.
    """
    def __init__(self, input_channels, color_channels=DATASET_CHANNELS):
        """
        Parameters
        ----------
        input_channels: int
            Number of channels of the input tensor.
        color_channels: int, options
            Number of channels in the color scheme of the trainig dataset.
            Default value is `DATASET_CHANNELS`.
        """
        super(ToColorChannels, self).__init__()
        self.conv = NormalConv2d(input_channels, color_channels, 1)

    def forward(self, x):
        """
        Outputs a tensor with a specified number of channels after applying a convolution to the input
        tensor `x`.

        Parameters
        ----------
        x: tensor
            The image tensor to be convoluted.

        Returns
        -------
        tensor
            The new tensor with a new number of channels resulting of convoluting the input.
        """
        return self.conv(x)
