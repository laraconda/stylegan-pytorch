import torch
from torch import nn
from models.utils import NormalConv2d, NormalLinear
from settings import BASE_DIM
from device import device


class AdaIn(nn.Module):
    def __init__(self, channels=BASE_DIM):
        super(AdaIn, self).__init__()
        self.inorm = nn.InstanceNorm2d(channels)
        self.scale = NormalLinear(BASE_DIM, channels)
        self.offset = NormalLinear(BASE_DIM, channels)

    def forward(self, x, w):
        normalizedx = self.inorm(x)
        scale = self.scale(w)
        offset = self.offset(w)
        scale = scale[:, :, None, None]
        offset = offset.unsqueeze(2).unsqueeze(3)
        return normalizedx * scale + offset


class Noise(nn.Module):
    def __init__(self, resolution, channels=BASE_DIM, *args, **kwargs):
        super(Noise, self).__init__(*args, **kwargs)
        self.weight = nn.Parameter(torch.randn((1, channels, 1, 1)))
        self.resolution = resolution

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, self.resolution, self.resolution), device=device)
        return x + noise * self.weight


class BasicSynthesisBlock(nn.Module):
    def __init__(self, resolution, input_channels, output_channels, *args, **kwargs):
        super(BasicSynthesisBlock, self).__init__(*args, **kwargs)
        self.conv1 = NormalConv2d(input_channels, input_channels, 3, padding='same')
        self.noise1 = Noise(resolution, input_channels)
        self.act = nn.LeakyReLU(0.2)
        self.adain1 = AdaIn(input_channels)
        self.conv2 = NormalConv2d(input_channels, output_channels, 3, padding='same')
        self.noise2 = Noise(resolution, output_channels)
        self.adain2 = AdaIn(output_channels)

    def forward(self, x, w):
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
    def __init__(self, resolution, channels=BASE_DIM):
        super(FirstSynthesisBlock, self).__init__()
        self.constant = nn.Parameter(
            torch.randn((1, channels, resolution, resolution))
        )
        self.basicblock = BasicSynthesisBlock(resolution, channels, channels)

    def forward(self, w):
        x = self.constant
        x = self.basicblock(x, w)
        return x


class SynthesisBlock(nn.Module):
    def __init__(self, resolution, input_channels, output_channels):
        super(SynthesisBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.basicblock = BasicSynthesisBlock(resolution, input_channels, output_channels)

    def forward(self, x, w):
        x = self.upsample(x)
        x = self.basicblock(x, w)
        return x
