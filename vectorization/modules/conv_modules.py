from itertools import islice

import torch.nn as nn
import torchvision.models as models

from vectorization.modules.maybe_module import MaybeModule
from .base import ParameterizedModule


class ResnetBlock(nn.Module):
    def __init__(self, resample=None):
        super(ResnetBlock, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.resample = resample

    def forward(self, x):
        identity = x

        out = self.conv(x)

        if self.resample is not None:
            identity = self.resample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(ResnetBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, resample=None, bn=True, expand=True):
        super(BasicBlock, self).__init__(resample=resample)

        self.conv = nn.Sequential(

            models.resnet.conv3x3(inplanes, planes, stride),
            MaybeModule(bn, nn.BatchNorm2d(planes)),
            nn.LeakyReLU(inplace=True),

            models.resnet.conv3x3(planes, planes),
            MaybeModule(bn, nn.BatchNorm2d(planes)),
        )


class Bottleneck(ResnetBlock):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, resample=None, bn=True, expand=True):
        super(Bottleneck, self).__init__(resample=resample)

        self.conv = nn.Sequential(

            models.resnet.conv1x1(inplanes, planes),
            MaybeModule(bn, nn.BatchNorm2d(planes)),
            nn.LeakyReLU(inplace=True),

            models.resnet.conv3x3(planes, planes, stride),
            MaybeModule(bn, nn.BatchNorm2d(planes)),
            nn.LeakyReLU(inplace=True),

            models.resnet.conv1x1(planes, planes * (self.expansion if expand else 1)),
            MaybeModule(bn, nn.BatchNorm2d(planes * (self.expansion if expand else 1))),
        )


resnet18 = (BasicBlock, [64, 128, 256, 512], [2, 2, 2, 2], [1, 2, 2, 2])
resnet34 = (BasicBlock, [64, 128, 256, 512], [3, 4, 6, 3], [1, 2, 2, 2])
resnet50 = (Bottleneck, [64, 128, 256, 512], [3, 4, 6, 3], [1, 2, 2, 2])
resnet101 = (Bottleneck, [64, 128, 256, 512], [3, 4, 23, 3], [1, 2, 2, 2])
resnet152 = (Bottleneck, [64, 128, 256, 512], [3, 8, 36, 3], [1, 2, 2, 2])


def resnet_model_creator(in_channels=1, convmap_channels=128, blocks=1, conf='18', bn=True):
    block_class, blocks_out_channels, blocks_in_layer, stride_in_layer = {
        '18': resnet18,
        '34': resnet34,
        '50': resnet50,
        '101': resnet101,
        '152': resnet152,
    }[conf]

    pre_channels = 64
    pre_layers = nn.Sequential(
        nn.Conv2d(in_channels, pre_channels, kernel_size=7, stride=2, padding=3, bias=False),
        MaybeModule(bn, nn.BatchNorm2d(pre_channels)),
        nn.LeakyReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    resnet_feat = [pre_layers]

    def _make_layer(block, out_channels, blocks, stride=1, in_channels=1, bn=True, convmap_channels=None):
        # first block should contain resampling layer if
        resample1 = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            resample1 = nn.Sequential(
                models.resnet.conv1x1(in_channels, out_channels * block.expansion, stride),
                MaybeModule(bn, nn.BatchNorm2d(out_channels * block.expansion)))
        layers = [block(in_channels, out_channels, stride, resample1, bn=bn)]

        # inbetween, "normal" blocks are inserted (no resampling needed)
        in_channels = out_channels * block_class.expansion
        for _ in range(1, blocks - 1):
            layers.append(block(in_channels, out_channels, bn=bn))

        # last block in layer, takes into account how many channels we want in the output
        resampleN = None
        if convmap_channels and convmap_channels != out_channels:
            out_channels = convmap_channels
            resampleN = nn.Sequential(
                models.resnet.conv1x1(in_channels, out_channels),
                MaybeModule(bn, nn.BatchNorm2d(out_channels)))
        layers.append(block(in_channels, out_channels, 1, resampleN, bn=bn, expand=False))

        return nn.Sequential(*layers)

    # Build the requested number of resnet blocks
    assert 1 <= blocks <= 4, 'number of blocks requested for ResNet model should be 1, 2, 3, or 4'

    in_channels = pre_channels
    for block_idx, out_channels, numblocks, stride in zip(
            range(0, blocks), blocks_out_channels, blocks_in_layer, stride_in_layer):

        convmap_channels_supplied = convmap_channels if block_idx == blocks - 1 else None
        layer = _make_layer(block_class, out_channels, numblocks, stride,
                            in_channels=in_channels, convmap_channels=convmap_channels_supplied, bn=bn)

        in_channels = out_channels * block_class.expansion
        resnet_feat.append(layer)

    return nn.Sequential(*resnet_feat)


def nth(l, x, n=1):
    """Returns n-th occurrence of x in l"""
    matches = (idx for idx, val in enumerate(l) if val == x)
    return next(islice(matches, n - 1, n), None)


def vgg_model_creator(in_channels=1, convmap_channels=128, blocks=1, conf='A', bn=False):
    # taken from source codes of vgg models: torchvision.models
    cfg = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }[conf]

    def _make_layers(cfg, batch_norm=False, in_channels=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    MaybeModule(batch_norm, nn.BatchNorm2d(v)),
                    nn.LeakyReLU(inplace=True)
                ]
                in_channels = v
        return nn.Sequential(*layers)

    last_pool_idx = nth(cfg, 'M', n=blocks)
    cfg[last_pool_idx - 1] = convmap_channels
    cfg = cfg[:last_pool_idx + 1]

    return _make_layers(cfg, batch_norm=bn, in_channels=in_channels)


class Conv(ParameterizedModule):
    def forward(self, images):
        return self.conv(images)


class ResnetConv(Conv):
    def __init__(self, in_channels=1, convmap_channels=128, blocks=1, conf=None, **kwargs):
        super().__init__(**kwargs)
        self.conv = resnet_model_creator(in_channels=in_channels, convmap_channels=convmap_channels,
                                         blocks=blocks, conf=conf)


class VggConv(Conv):
    def __init__(self, in_channels=1, convmap_channels=128, blocks=1, conf=None, bn=True, **kwargs):
        super().__init__(**kwargs)
        self.conv = vgg_model_creator(in_channels=in_channels, convmap_channels=convmap_channels,
                                      blocks=blocks, conf=conf, bn=bn)


conv_module_by_kind = {
    'resnet': ResnetConv,
    'vgg': VggConv,
}


__all__ = [
    'conv_module_by_kind'
]
