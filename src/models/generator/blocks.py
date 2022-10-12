"""
Definition of different blocks for the networks that compose StyleGAN.
"""

import torch
from torch import nn
from models.utils import NormalConv2d, NormalLinear
from settings import BASE_DIM
from device import device


class AdaIn(nn.Module):
    """
    AdaIn block. Performs the AdaIN operation on a tensor input.

    Attributes
    ----------
    inorm: InstanceNorm2d
        An instance normalization instance.
    scale: NormalLinear
        A dense layer used to scale the input tensor.
    offset: NormalLinear
        A dense layer used to offset the input tensor.

    Methods
    -------
    forward(x, w)
        Performs the AdaIN operation on the `x` tensor based on the tensor `w`.
    """
    def __init__(self, channels=BASE_DIM):
        """
        Parameters
        ----------
        channels: int
            Number of channels of the input and output.
        """
        super(AdaIn, self).__init__()
        self.inorm = nn.InstanceNorm2d(channels)
        self.scale = NormalLinear(BASE_DIM, channels)
        self.offset = NormalLinear(BASE_DIM, channels)

    def forward(self, x, w):
        """
        Performs the AdaIN operation on the x tensor based on the tensor `w`.

        Parameters
        ----------
        x: tensor
            The tensor to be transformed.
        w: tensor
            The 'style' tensor where the variables scale and offset
            come from after two different linear transformations on `w`.

        Returns
        -------
        tensor
            The result of the transformation.
        """
        normalizedx = self.inorm(x)
        scale = self.scale(w)
        offset = self.offset(w)
        scale = scale[:, :, None, None]
        offset = offset.unsqueeze(2).unsqueeze(3)
        return normalizedx * scale + offset


class Noise(nn.Module):
    """
    Noise layer. Injects random noise into an input tensor.

    Attributes
    ----------
    weight: Parameter
        A learnable parameter used to scale a random tensor.
    resolution: int
        Resolution of the image tensors the layer is going to process.

    Methods
    -------
    forward(x)
        Injects random noise into the input tensor x.
    """
    def __init__(self, resolution, channels=BASE_DIM, *args, **kwargs):
        """
        Parameters
        ----------
        resolution: int
            The image resolution on which this instance is going to operate.
        channels: int
            Number of channels of this instance.
        """
        super(Noise, self).__init__(*args, **kwargs)
        self.weight = nn.Parameter(torch.randn((1, channels, 1, 1)))
        self.resolution = resolution

    def forward(self, x):
        """
        Injects scales a random tensor and injects it into the input tensor `x`.

        Parameters
        ----------
        x: tensor
            The tensor on which the transformation of injecting noise is going to be applied.

        Returns
        -------
        tensor
            The result of the transformation.
        """
        noise = torch.randn((x.shape[0], 1, self.resolution, self.resolution), device=device)
        return x + noise * self.weight


class BasicSynthesisBlock(nn.Module):
    """
    Defines a series of layers common for all synthesis blocks.

    Attributes
    ----------
    conv1: NormalConv2d
        First convolutional layer.
    noise1: Noise
        First noise layer.
    act: LeakyReLU
        Activation. ReLU with a slope.
    adain1: AdaIn
        First adain layer.
    conv2: NormalConv2d
        Second convolution layer.
    noise2: Noise
        Second noise layer.
    adain2: AdaIn
        Second AdaIN layer.

    Methods
    -------
    forward(x, w)
        Feeds forward the input tensor `x` through the layers.
    """
    def __init__(self, resolution, input_channels, output_channels):
        """
        Parameters
        ----------
        resolution: int
            Resolution of the image tensors the block is going to process.
        input_channels: int
            Number of channels of the input.
        output_channels:
            Number of output channels of the output.
        """
        super(BasicSynthesisBlock, self).__init__()
        self.conv1 = NormalConv2d(input_channels, input_channels, 3, padding='same')
        self.noise1 = Noise(resolution, input_channels)
        self.act = nn.LeakyReLU(0.2)
        self.adain1 = AdaIn(input_channels)
        self.conv2 = NormalConv2d(input_channels, output_channels, 3, padding='same')
        self.noise2 = Noise(resolution, output_channels)
        self.adain2 = AdaIn(output_channels)

    def forward(self, x, w):
        """
        Feeds forward the input tensor `x` through the layers.

        Parameters
        ----------
        x: tensor
            The tensor to be feed forwarded.

        w: tensor
            The 'style' tensor. AdaIN transformations use it to transform the `x` tensor.

        Returns
        -------
        tensor
            The result of the transformations.
        """
        x = self.conv1(x)
        x = self.noise1(x)
        x = self.act(x)
        x = self.adain1(x, w)
        x = self.conv2(x)
        x = self.noise2(x)
        x = self.act(x)
        x = self.adain2(x, w)
        return x


class FirstSynthesisBlock(nn.Module):
    """
    The first block of a synthesis network.

    Attributes
    ----------
    constant: Parameter
        Learnable parameter. The very beginning of the block an thus of a network.
    basicblock: BasicSynthesisBlock
        A basic synthesis block.

    Methods
    -------
    forward(w)
        Given a style `w`, it returns a tensor after transforming the constant using the
        basic block.
    """
    def __init__(self, resolution, channels=BASE_DIM):
        """
        Parameters
        ----------
        resolution: int
            Resolution of the image tensors the block is going to process.

        channels: int
            Number of channels of the output of this block.

        """
        super(FirstSynthesisBlock, self).__init__()
        self.constant = nn.Parameter(
            torch.randn((1, channels, resolution, resolution))
        )
        self.basicblock = BasicSynthesisBlock(resolution, channels, channels)

    def forward(self, w):
        """
        Given a style `w`, it returns a tensor after transforming the constant using the
        basic block.

        Parameters
        ----------
        w: tensor
            Style tensor.

        Returns
        -------
        tensor
            The constant tensor transformed by the operations defined by the basic block.
        """
        x = self.constant
        x = self.basicblock(x, w)
        return x


class SynthesisBlock(nn.Module):
    """
    Defines a synthesis block (other than the first one).

    Attributes
    ----------
    upsample: Upsample
        layer used to upsample a tensor input.
    basicblock: BasicSynthesisBlock
        A basic sinthesis block.

    Methods
    -------
    forward(x, w)
        Upsamples the tensor `x` and transforms it.
    """
    def __init__(self, resolution, input_channels, output_channels):
        """
        Parameters
        ----------
        resolution: int
            Resolution of the image tensors this block is going to process.
        input_channels: int
            Number channels in the input.
        output_channels: int
            Number of channels in the output of this block.
        """
        super(SynthesisBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.basicblock = BasicSynthesisBlock(resolution, input_channels, output_channels)

    def forward(self, x, w):
        """
        Upsamples the tensor `x` and transforms it based on `w`.

        Parameters
        ----------
        x: tensor
            An image tensor to be transformed.
        w: tensor
            A style tensor.

        Returns
        -------
        tensor
            The result of the transformations applied to `x`.
        """
        x = self.upsample(x)
        x = self.basicblock(x, w)
        return x
