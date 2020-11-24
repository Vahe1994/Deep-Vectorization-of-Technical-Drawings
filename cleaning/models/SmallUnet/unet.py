# code provided by Ravil Khisamov

import os
import gc

import torch
import torch, torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import numpy as np


def CreateConvBnRelu(in_channels, out_channels, dilation=1):
    module = nn.Sequential()
    module.add_module('conv', nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        bias=False,
        dilation=dilation,
        padding=dilation))
    module.add_module('bn', nn.BatchNorm2d(
        num_features=out_channels))
    module.add_module('relu', nn.ReLU())
    return module


class SmallUnet(nn.Module):
    def __init__(self):
        super(SmallUnet, self).__init__()

        # encoder
        self.add_module('EncConvBnRelu1_1', CreateConvBnRelu(3, 64))
        self.add_module('EncConvBnRelu1_2', CreateConvBnRelu(64, 64))
        self.add_module('EncMp1', nn.MaxPool2d(kernel_size=2))

        self.add_module('EncConvBnRelu2_1', CreateConvBnRelu(64, 128))
        self.add_module('EncConvBnRelu2_2', CreateConvBnRelu(128, 128))
        self.add_module('EncMp2', nn.MaxPool2d(kernel_size=2))

        self.add_module('EncConvBnRelu3_1', CreateConvBnRelu(128, 256))
        self.add_module('EncConvBnRelu3_2', CreateConvBnRelu(256, 256))
        self.add_module('EncMp3', nn.MaxPool2d(kernel_size=2))

        # lowest layer
        self.add_module('ConvBnRelu4_1', CreateConvBnRelu(256, 512))
        self.add_module('ConvBnRelu4_2', CreateConvBnRelu(512, 512))
        self.add_module('Us4', nn.Upsample(scale_factor=2))

        # decoder
        self.add_module('DecConvBnRelu3_1', CreateConvBnRelu(512 + 256, 256))
        self.add_module('DecConvBnRelu3_2', CreateConvBnRelu(256, 256))
        self.add_module('DecUs3', nn.Upsample(scale_factor=2))

        self.add_module('DecConvBnRelu2_1', CreateConvBnRelu(256 + 128, 128))
        self.add_module('DecConvBnRelu2_2', CreateConvBnRelu(128, 128))
        self.add_module('DecUs2', nn.Upsample(scale_factor=2))

        # prediction
        self.add_module('PredConvBnRelu_1', CreateConvBnRelu(128 + 64, 64))
        self.add_module('PredConvBnRelu_2', CreateConvBnRelu(64, 64))
        self.add_module('PredDense', nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=5,
            padding=2
        ))
        self.add_module('PredProbs', nn.Sigmoid())

    def forward(self, x):
        enc_1 = self.EncConvBnRelu1_2(self.EncConvBnRelu1_1(x))
        x = self.EncMp1(enc_1)

        enc_2 = self.EncConvBnRelu2_2(self.EncConvBnRelu2_1(x))
        x = self.EncMp2(enc_2)

        enc_3 = self.EncConvBnRelu3_2(self.EncConvBnRelu3_1(x))
        x = self.EncMp2(enc_3)

        x = self.ConvBnRelu4_2(self.ConvBnRelu4_1(x))
        x = self.Us4(x)

        x = torch.cat((x, enc_3), dim=1)
        x = self.DecConvBnRelu3_2(self.DecConvBnRelu3_1(x))
        x = self.DecUs3(x)

        x = torch.cat((x, enc_2), dim=1)
        x = self.DecConvBnRelu2_2(self.DecConvBnRelu2_1(x))
        x = self.DecUs2(x)

        x = torch.cat((x, enc_1), dim=1)
        x = self.PredConvBnRelu_2(self.PredConvBnRelu_1(x))
        x = self.PredDense(x)
        x = self.PredProbs(x)

        x = x.squeeze(dim=1)

        return x
