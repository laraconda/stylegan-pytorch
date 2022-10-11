"""
Declaration of blocks for the discriminator network and declaration of the network itself.
"""

from math import log2
from torch import nn
import torch
import torch.nn.functional as F
from src.models.utils import NormalConv2d, NormalLinear, FromRGB


class DiscriminatorBlock(nn.Module):
    """
    Defines a series of layers that represent a block of the discriminator.

    Attributes
    ----------
    conv1: NormalConv2d
        First convolutional layer.
    conv2: NormalConv2d
        Second convolutional layer.
    act: LeakyReLU
        Activation.
    downsample: Upsample
        Upsample instance with a scale factor of 0.5.

    Methods
    -------
    forward(x)
        Receives a tensor image and downsamples it after treating it with convolutional
        layers and activations.

    """
    def __init__(self, input_channels, output_channels):
        """
        Parameters
        ----------
        input_channels: int
            Number of channels in the input of this block.
        output_channels: int
            Number of channels of the output of this block.
        """
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = NormalConv2d(input_channels, input_channels, 3, padding='same')
        self.conv2 = NormalConv2d(input_channels, output_channels, 3, padding='same')
        self.act = nn.LeakyReLU(0.2)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear')

    def forward(self, x):
        """
        Receives a tensor image and downsamples it after treating it with convolutional
        layers and activations.

        Parameters
        ----------
        x: tensor
            Tensor to be treated and downsampled.

        Returns
        -------
        tensor
            The processed and downsampled input tensor.
        """
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return self.downsample(x)


class DiscriminatorFinalBlock(nn.Module):
    """
    Final block of the discriminator network.

    Attributes
    ----------
    conv1: NormalConv2d
        First convolutional layer.
    conv2: NormalConv2d
        Second convolutional layer.
    act: LeakyReLU
        Activation.
    fc: NormalLinear
        Dense layer with a single output feature.

    Methods
    -------
    forward(x)
        Outputs a single value representing the critic score to the input.
    """
    def __init__(self, input_channels):
        """
        Parameters
        ----------
        input_channels: int
            Number of channels of the input.
        """
        super(DiscriminatorFinalBlock, self).__init__()
        self.conv1 = NormalConv2d(input_channels + 1, input_channels, 3, padding='same')
        self.conv2 = NormalConv2d(input_channels, input_channels, 4)
        self.act = nn.LeakyReLU(0.2)
        self.fc = NormalLinear(input_channels, 1)

    def _minibatch_stddev(self, x):
        """
        Performs the minibatch standard deviation operation on the input `x`.

        Parameters
        ----------
        x: tensor
            Input on which the operation is applied, it's resolution should be 4 by 4.

        Returns
        -------
        tensor
            The result of the minibatch standard deviation operation on the input.
        """
        var = x.var(0, unbiased=False) + 1e-8  # Avoid zero
        std = torch.sqrt(var)
        mean_std = std.mean().expand(x.size(0), 1, 4, 4)
        x = torch.cat([x, mean_std], 1)
        return x

    def forward(self, x):
        """
        Parameters
        ----------
        x: tensor
            Input to be judged by this block, the last stage of the critic.
            It's resolution should be 4 by 4.

        Returns
        -------
        int
            A single value representing the critic score to the input.
        """
        x = self._minibatch_stddev(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = x.transpose(1, 3)
        x = self.fc(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator Network.

    Attributes
    ----------
    alpha: int
        Alpha step of the training, if the network is not being trained, the value
        should be set to 1.
    start_at_resolution: int
        The resolution of the input images to be critiqued.
    blocks: ModuleList
        List of discriminator blocks to process different resolution of image tensors.
    from_rgb: ModuleList
        List of FromRGB instances to convert 3-channeled RGB image tensors into
        multi-channeled image tensors.

    Methods
    -------
    forward(x)
         Returns a single value representing the critic score to the input `x`.
    blocktores(block_id)
        Given the id of a block, it returns the resolution at which said block operates.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.alpha = 1
        self.start_at_resolution = 1024
        self.blocks = nn.ModuleList([    # resolution, block_id
            DiscriminatorBlock(16, 32),    # 1024,       0
            DiscriminatorBlock(32, 64),    # 512,        1
            DiscriminatorBlock(64, 128),   # 256,        2
            DiscriminatorBlock(128, 256),  # 128,        3
            DiscriminatorBlock(256, 512),  # 64,         4
            DiscriminatorBlock(512, 512),  # 32,         5
            DiscriminatorBlock(512, 512),  # 16,         6
            DiscriminatorBlock(512, 512),  # 8,          7
            DiscriminatorFinalBlock(512),  # 4,          8
        ])

        self.from_rgb = nn.ModuleList([
            FromRGB(16),
            FromRGB(32),
            FromRGB(64),
            FromRGB(128),
            FromRGB(256),
            FromRGB(512),
            FromRGB(512),
            FromRGB(512),
            FromRGB(512)
        ])

    def _downsample_half(self, image):
        """
        Resizes in half a tensor image using a bilinear interpolation.

        Parameters
        ----------
        image: tensor
            Tensor image to be resized in half.

        Returns
        -------
        tensor
            The resized image.
        """
        return F.interpolate(image, scale_factor=0.5, mode='bilinear')

    def _combine_images(self, image_a, image_b, alpha):
        """
        Combines `image_a` and `image_b` into one image, with the level
        of influence of `image_a` over the final image determined by `alpha`.

        Parameters
        ----------
            image_a: tensor
                The first image to be combined.
            image_b: tensor
                The second image to be combined.
            alpha: float
                The degree of influence of `image_a` over the final image.
                Goes from 0 to 1.

        Returns
        -------
        tensor
            The image resulting from combining the two images, given alpha.
        """
        return torch.lerp(image_a, image_b, alpha)

    def forward(self, x):
        """
        Returns a single value representing the critic score to the input `x`.

        Parameters
        ----------
        x: tensor
            Input to be judged by the network.

        Returns
        -------
        int
            A single value representing the critic score to the input.
        """
        start_at_block = self._restoblock(self.start_at_resolution)
        downsampled_image = self.from_rgb[start_at_block + 1](self._downsample_half(x))
        conv_image = self.blocks[start_at_block](self.from_rgb[start_at_block](x))
        x = self._combine_images(downsampled_image, conv_image, self.alpha)
        for i in range(start_at_block + 1, len(self.blocks)):
            x = self.blocks[i](x)
        return torch.squeeze(x)

    @classmethod
    def _restoblock(cls, res):
        """
        Returns the block id of the block responsible for handling the resolution `res`.

        Parameters
        ----------
        res: int
            An image resolution, it must be a power of 2.

        Returns
        -------
        int
            The id (based on the attribute 'blocks') of the block that corresponds to
            the resolution `res`.

        Raises
        ------
        AssertionError
            If the image resolution is not a power of two.
        """
        log2res = log2(res)
        assert log2res.is_integer(), "resolution should be a power of 2."
        block_id = 10 - int(log2res)
        return block_id

    @classmethod
    def blocktores(cls, block_id):
        """
        Returns the resolution that corresponds to the block identified by `block_id`.

        Parameters
        ----------
        block_id: int
            The identifier of the block whose resolution is needed.

        Returns
        -------
        int
            The resolution that corresponds to the block identified by `block_id`.
            It is always a power of 2.

        Raises
        ------
        AssertionError
            If 'block_id' >= 9 (the number of blocks).
        """
        assert block_id < 9, "block id can't be 9 or more."
        return 2**(10 - block_id)
