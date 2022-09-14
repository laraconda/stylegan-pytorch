from math import log2
from torch import nn
import torch
import torch.nn.functional as F
from models.utils import NormalConv2d, NormalLinear, FromRGB


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, *args, **kwargs):
        super(DiscriminatorBlock, self).__init__(*args, **kwargs)
        self.conv1 = NormalConv2d(input_channels, input_channels, 3, padding='same')
        self.conv2 = NormalConv2d(input_channels, output_channels, 3, padding='same')
        self.act = nn.LeakyReLU(0.2)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return self.downsample(x)


class DiscriminatorFinalBlock(nn.Module):
    def __init__(self, input_channels):
        super(DiscriminatorFinalBlock, self).__init__()
        self.conv1 = NormalConv2d(input_channels + 1, input_channels, 3, padding='same')
        self.conv2 = NormalConv2d(input_channels, input_channels, 4)
        self.act = nn.LeakyReLU(0.2)
        self.fc = NormalLinear(input_channels, 1)
        self.sgmd = nn.Sigmoid()

    def minibatch_stddev(self, x):
        var = x.var(0, unbiased=False) + 1e-8  # Avoid zero
        std = torch.sqrt(var)
        mean_std = std.mean().expand(x.size(0), 1, 4, 4)
        x = torch.cat([x, mean_std], 1)
        return x

    def forward(self, x):
        x = self.minibatch_stddev(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = x.transpose(1, 3)
        x = self.fc(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
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

    def downsampleHalf(self, image):
        return F.interpolate(image, scale_factor=0.5, mode='bilinear')

    def combine_images(self, image_a, image_b, alpha):
        return torch.lerp(image_a, image_b, alpha)

    def forward(self, x):
        start_at_block = self.restoblock(self.start_at_resolution)
        downsampled_image = self.from_rgb[start_at_block + 1](self.downsampleHalf(x))
        conv_image = self.blocks[start_at_block](self.from_rgb[start_at_block](x))
        x = self.combine_images(downsampled_image, conv_image, self.alpha)
        for i in range(start_at_block + 1, len(self.blocks)):
            x = self.blocks[i](x)
        return torch.squeeze(x)

    @classmethod
    def restoblock(cls, res):
        log2res = log2(res)
        assert log2res.is_integer(), "resolution should be a power of 2."
        block_id = 10 - int(log2res)
        return block_id

    @classmethod
    def blocktores(cls, block_id):
        assert block_id < 9, "block id can't be 9 or more."
        return 2**(10 - block_id)
