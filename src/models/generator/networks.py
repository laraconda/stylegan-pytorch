from math import log2
from torch import nn
import torch
import random
import torch.nn.functional as F
from src.models.generator.blocks import FirstSynthesisBlock, SynthesisBlock
from src.models.utils import NormalLinear, ToRGB
from src.settings import BASE_DIM


class MappingNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MappingNetwork, self).__init__(*args, **kwargs)
        self.fcs = nn.Sequential(
            NormalLinear(BASE_DIM, BASE_DIM),
            nn.ReLU(),
            NormalLinear(BASE_DIM, BASE_DIM),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fcs(x)


class SynthesisNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SynthesisNetwork, self).__init__(*args, **kwargs)
        self.conv_blocks = nn.ModuleList([                   # resolution, block_id
            FirstSynthesisBlock(4),                            # 4,          0
            SynthesisBlock(8, BASE_DIM, BASE_DIM),             # 8,          1
            SynthesisBlock(16, BASE_DIM, BASE_DIM),            # 16,         2
            SynthesisBlock(32, BASE_DIM, BASE_DIM),            # 32,         3
            SynthesisBlock(64, BASE_DIM, BASE_DIM // 2),         # 64,         4
            SynthesisBlock(128, BASE_DIM // 2, BASE_DIM // 4),     # 128,        5
            SynthesisBlock(256, BASE_DIM // 4, BASE_DIM // 8),     # 256,        6
            SynthesisBlock(512, BASE_DIM // 8, BASE_DIM // 16),    # 512,        7
            SynthesisBlock(1024, BASE_DIM // 16, BASE_DIM // 32),  # 1024,       8
        ])

        self.to_rgb = nn.ModuleList([
            ToRGB(BASE_DIM),
            ToRGB(BASE_DIM),
            ToRGB(BASE_DIM),
            ToRGB(BASE_DIM),
            ToRGB(BASE_DIM // 2),
            ToRGB(BASE_DIM // 4),
            ToRGB(BASE_DIM // 8),
            ToRGB(BASE_DIM // 16),
            ToRGB(BASE_DIM // 32),
        ])

    def upsample2x(self, image):
        return F.interpolate(image, scale_factor=2, mode='bilinear')

    def combine_images(self, image_a, image_b, alpha):
        return torch.lerp(image_a, image_b, alpha)

    @classmethod
    def restoblock(cls, res):
        log2res = log2(res)
        assert log2res.is_integer(), "stop_at_resolution should be a power of 2."
        block_id = int(log2res) - 2
        return block_id

    @classmethod
    def blocktores(cls, block_id):
        return 2**(block_id + 2)

    def block_sequential_exec(self, w1, w2, alpha, stop_at_resolution, style_mixing):
        stop_at_block = self.restoblock(stop_at_resolution)
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
        upsample_image = self.upsample2x(image_to_upsample)
        conv_image = self.to_rgb[stop_at_block](self.conv_blocks[stop_at_block](x, w))
        x = self.combine_images(upsample_image, conv_image, alpha)
        return x

    def forward(self, w1, w2, alpha=1, stop_at_resolution=1024, style_mixing=False):
        return self.block_sequential_exec(w1, w2, alpha, stop_at_resolution, style_mixing)


class StyleGAN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(StyleGAN, self).__init__(*args, **kwargs)
        self.mappn = MappingNetwork()
        self.synthn = SynthesisNetwork()
        self.prob_style_mixing = 0.3
        self.stop_at_resolution = 1024
        self.alpha = 1

    def forward(self, z1, z2=None):
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
        return SynthesisNetwork.blocktores(block_id)
