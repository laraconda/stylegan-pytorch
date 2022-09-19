from torch import nn
from src.settings import STD


class NormalConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(NormalConv2d, self).__init__(*args, **kwargs)
        self.weight.data.normal_(std=STD)
        self.bias.data.zero_()


class NormalLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(NormalLinear, self).__init__(*args, **kwargs)
        self.weight.data.normal_(std=STD)
        self.bias.data.zero_()


class FromRGB(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super(FromRGB, self).__init__(*args, **kwargs)
        self.conv = NormalConv2d(3, channels, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        return self.act(x)


class ToRGB(nn.Module):
    def __init__(self, input_channels, *args, **kwargs):
        super(ToRGB, self).__init__(*args, **kwargs)
        self.conv = NormalConv2d(input_channels, 3, 1)

    def forward(self, x):
        return self.conv(x)
