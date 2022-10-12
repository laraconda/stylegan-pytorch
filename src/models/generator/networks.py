"""
Definition of networks that compose the StyleGAN generator network.
"""

from math import log2
from torch import nn
import torch
import random
import torch.nn.functional as F
from models.generator.blocks import FirstSynthesisBlock, SynthesisBlock
from models.utils import NormalLinear, ToColorChannels
from settings import BASE_DIM


class MappingNetwork(nn.Module):
    """
    Mapping Network of the StyleGAN architecture.

    Maps a z tensor into a w tensor.

    Attributes
    ----------
    fcs: Sequential
        A sequence of dense layers.

    Methods
    -------
    forward(z)
        Passes the `z` tensor through the sequence `fcs`.
    """
    def __init__(self):
        super(MappingNetwork, self).__init__()
        self.fcs = nn.Sequential(
            NormalLinear(BASE_DIM, BASE_DIM),
            nn.ReLU(),
            NormalLinear(BASE_DIM, BASE_DIM),
            nn.ReLU()
        )

    def forward(self, z):
        """
        Passes the `z` tensor through the sequence `fcs`.

        Parameters
        ----------
        z: tensor
            The tensor to be mapped.

        Returns
        -------
        tensor
            The result of the mapping, a w tensor.
        """
        return self.fcs(z)


class SynthesisNetwork(nn.Module):
    """
    Synthesis Network of the StyleGAN architecture.
    Progressively grows an image with the help of a set of w styles.

    Attributes
    ----------
    conv_blocks: ModuleList
        A list of synthesis blocks.
    to_rgb: ModuleList
        A list of ToColorChannels layers.

    Methods
    -------
    forward(z, w1, w2, alpha=1, stop_at_resolution=1024, style_mixing=False)
        Produces an output image of resolution `stop_at_resolution`.
    blocktores(block_id)
        Returns the resolution that corresponds to the block identified by `block_id`.
    """
    def __init__(self):
        super(SynthesisNetwork, self).__init__()
        self.conv_blocks = nn.ModuleList([                         # res,  block_id
            FirstSynthesisBlock(4),                                # 4,    0
            SynthesisBlock(8, BASE_DIM, BASE_DIM),                 # 8,    1
            SynthesisBlock(16, BASE_DIM, BASE_DIM),                # 16,   2
            SynthesisBlock(32, BASE_DIM, BASE_DIM),                # 32,   3
            SynthesisBlock(64, BASE_DIM, BASE_DIM // 2),           # 64,   4
            SynthesisBlock(128, BASE_DIM // 2, BASE_DIM // 4),     # 128,  5
            SynthesisBlock(256, BASE_DIM // 4, BASE_DIM // 8),     # 256,  6
            SynthesisBlock(512, BASE_DIM // 8, BASE_DIM // 16),    # 512,  7
            SynthesisBlock(1024, BASE_DIM // 16, BASE_DIM // 32),  # 1024, 8
        ])

        self.to_rgb = nn.ModuleList([
            ToColorChannels(BASE_DIM),
            ToColorChannels(BASE_DIM),
            ToColorChannels(BASE_DIM),
            ToColorChannels(BASE_DIM),
            ToColorChannels(BASE_DIM // 2),
            ToColorChannels(BASE_DIM // 4),
            ToColorChannels(BASE_DIM // 8),
            ToColorChannels(BASE_DIM // 16),
            ToColorChannels(BASE_DIM // 32),
        ])

    def _upsample2x(self, image):
        """
        Doubles the resolution of `image` using a bilinear interpolation.

        Parameters
        ----------
        image: tensor
            Image to upsample.

        Returns
        -------
        tensor
            The upsampled image.
        """
        return F.interpolate(image, scale_factor=2, mode='bilinear')

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
            The id (based on the attribute `conv_blocks`) of the block that
            corresponds to the resolution `res`.

        Raises
        ------
        AssertionError
            If the image resolution is not a power of two.
        """
        log2res = log2(res)
        assert log2res.is_integer(), "stop_at_resolution should be a power of 2."
        block_id = int(log2res) - 2
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
        """
        return 2**(block_id + 2)

    def _block_sequential_exec(self, w1, w2, alpha, stop_at_resolution, style_mixing):
        """
        Executes the blocks in `conv_blocks` in a sequential order thus growing an image
        as the sequence progresses (progressive growing).

        Parameters
        ----------
        w1: tensor
            The first style tensor.
        w2: tensor
            The second style tensor, used for style mixing.
        alpha: float
            When combining two images, one being the result of a block and the
            other being the result of an algorithmic upsample, it determines the
            influence of one over the other.
        stop_at_resolution: int
            The progressive growing stops at the resolution set by this parameter.
        style_mixing: bool
            Controls wheter or not at some random point of the progressive growing
            the second style tensor `w2` replaces the first one `w1`.

        Returns
        -------
        tensor
            An image resulting from the progressive growing of the synthesis network.
            Its resolution is dictated by `stop_at_resolution`.
        """
        stop_at_block = self._restoblock(stop_at_resolution)
        x = self.conv_blocks[0](w1)
        if stop_at_block == 0:
            return self.to_rgb[stop_at_block](x)
        w = w1
        if style_mixing:
            style_mixing_point = random.randint(0, stop_at_block - 1)
        for i in range(1, stop_at_block):
            if style_mixing and style_mixing_point == i:
                w = w2
            x = self.conv_blocks[i](x, w)
        image_to_upsample = self.to_rgb[stop_at_block - 1](x)
        upsample_image = self._upsample2x(image_to_upsample)
        conv_image = self.to_rgb[stop_at_block](self.conv_blocks[stop_at_block](x, w))
        x = self._combine_images(upsample_image, conv_image, alpha)
        return x

    def forward(self, w1, w2, alpha=1, stop_at_resolution=1024, style_mixing=False):
        """
        Public method that wraps the `_block_sequential_exec` private method.

        Parameters
        ----------
        w1: tensor
            The first style tensor.
        w2: tensor
            The second style tensor, used for style mixing.
        alpha: float, optional
            When combining two images, one being the result of a block and the
            other being the result of an algorithmic upsample, it determines the
            influence of one over the other (default is 1).
        stop_at_resolution: int, optional
            The progressive growing stops at the resolution set by this parameter
            (default is 1024).
        style_mixing: bool, optional
            Controls wheter or not at some random point of the progressive growing
            the second style tensor `w2` replaces the first one `w1` (default is False).

        Returns
        -------
        tensor
            An image resulting from the progressive growing of the synthesis network.
            Its resolution is dictated by `stop_at_resolution`.
        """
        return self._block_sequential_exec(w1, w2, alpha, stop_at_resolution, style_mixing)


class StyleGAN(nn.Module):
    """
    Generator of the StyleGAN architecture.
    Generates an output image out of a set of random z tensors.

    Attributes
    ----------
    mappn: MappingNetwork
        Instance of the MappingNetowrk class. It is the first sub-network of this network.
    synthn: SynthesisNetwork
        Instance of the SynthesisNetwork class. It's the second sub-network of this network.
    prob_style_mixing: float
        The probability that the style mixing trick will take effect during a fordward
        pass of the network.
    stop_at_resolution: int
        The resolution of the resulting image tensor.
    alpha: float
        The alpha step of the training, if the instance of the network is not being trained,
        the value should be 1.

    Methods
    -------
    forward(z1, z2=None)
        Returns an output image tensor generated by the network.
    blocktores(block_id)
        Returns the resolution that corresponds to the block identified by `block_id`.
    """
    def __init__(self):
        super(StyleGAN, self).__init__()
        self.mappn = MappingNetwork()
        self.synthn = SynthesisNetwork()
        self.prob_style_mixing = 0.3
        self.stop_at_resolution = 1024
        self.alpha = 1

    def forward(self, z1, z2=None):
        """
        Returns an output image tensor generated by the network.

        Parameters
        ----------
        z1: tensor
            A random tensor used as a seed to produce a tensor image.
        z2: tensor, optional
            A random tensor used as a seed to produce a tensor image.
            If its None then it won't be used and the style mixing effect won't
            take place (default is None).

        Returns
        -------
        tensor
            Returns an output image tensor generated by the network.
        """
        w1 = self.mappn(z1)
        if z2 is not None:
            w2 = self.mappn(z2)
            style_mixing = random.random() <= self.prob_style_mixing
        else:
            w2 = None
            style_mixing = False
        x = self.synthn(w1, w2, self.alpha, self.stop_at_resolution, style_mixing)
        return x

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
        """
        return SynthesisNetwork.blocktores(block_id)
