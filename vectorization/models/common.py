import torch
import torch.nn as nn
import torchvision.models as models


class SpecifiedModuleBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def from_model_spec(cls, spec):
        return cls(**spec)


def resnet_model_creator(in_channels=1, convmap_channels=128, blocks=1, conf='18'):
    cfg = {
        '18': models.resnet18,
        '34': models.resnet34,
        '50': models.resnet50,
        '101': models.resnet101,
        '152': models.resnet152,
    }[conf]
    resnet = cfg()
    resnet.conv1.in_channels = in_channels

    assert 1 <= blocks <= 4, 'number of blocks requested for ResNet model should be 1, 2, 3, or 4'
    resnet_feat = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool]
    resnet_layers = [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
    for _, layer in zip(range(0, blocks), resnet_layers):
        resnet_feat.append(layer)
    # we assume (1) each layer is at least two Sequentials
    last_block = layer[-1]
    if isinstance(last_block, models.resnet.BasicBlock):
        last_block.conv2.out_channels = convmap_channels
        last_block.bn2.in_channels = convmap_channels
    else:
        assert isinstance(last_block, models.resnet.Bottleneck)
        last_block.conv3.out_channels = convmap_channels
        last_block.bn3.in_channels = convmap_channels

    return nn.Sequential(*resnet_feat)


def vgg_model_creator(in_channels=1, convmap_channels=128, blocks=1, conf='A', bn=False):
    # taken from source codes of vgg models: torchvision.models
    cfg = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }[conf]

    def _make_layers(cfg, batch_norm=False, in_channels=1, convmap_channels=128, blocks=1):
        layers = []
        block_idx = 0  # how many max-pool layers have passed
        for v in cfg:
            if block_idx == blocks:
                # this should be true as block_idx can be incremented only after creating 'M' layer,
                # which is done only after at least one conv2d is created
                assert None is not conv2d
                conv2d.out_channels = convmap_channels
                break

            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                block_idx += 1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.LeakyReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    return _make_layers(cfg, batch_norm=bn, in_channels=in_channels,
                        convmap_channels=convmap_channels, blocks=blocks)


model_creator = {
    'resnet': resnet_model_creator,
    'vgg': vgg_model_creator,
}


class ConvFeatureExtractor(SpecifiedModuleBase):
    def __init__(self, in_channels=1, convmap_channels=128, kind='resnet', blocks=1, conf=None, **kwargs):
        """

        :param in_channels: number of channels in source image (typically 1 or 3)
        :param hidden_dim:
        :param resnet_count:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.conv = model_creator[kind](input_channels=in_channels, convmap_channels=convmap_channels,
                                        blocks=blocks, conf=conf, **kwargs)

    def forward(self, images):
        return self.conv(images)


class Hidden(SpecifiedModuleBase):
    def forward(self, conv_features, max_lines):
        pass


class VectorizationOutput(SpecifiedModuleBase):
    def __init__(self, hidden_dim=128, ffn_dim=512, n_head=8, num_layers=10, input_channels=1, output_dim=5,
                 resnet_count=0, **kwargs):
        super().__init__(**kwargs)
        self.final_fc = nn.Linear(hidden_dim, self.output_dim)
        self.final_tanh= nn.Tanh()
        self.final_sigm = nn.Sigmoid()
        self.relu  = nn.ReLU()

    def forward(self, ):
        fc = self.final_fc(self.relu(h_dec)) #[b, n, output_dim]
        coord = (self.final_tanh(fc[:,:,:-1]) + 1.) / 2. #[b, n, output_dim-1]
        prob = self.final_sigm(fc[:,:,-1]).unsqueeze(-1)
        return torch.cat((coord,prob), dim = -1) #[b, n, output_dim]


class VectorizationModelBase(SpecifiedModuleBase):
    def __init__(self, features, hidden, output):
        """
        :param input_channels: number of input channels in image
        :param convmap_channels: number of convolutional feature channels extracted from image
        """
        super().__init__()
        self.features = features
        self.hidden = hidden
        self.output = output

    def forward(self, images, n):
        x = self.features(images)
        x = self.hidden(x, n)
        x = self.output(x)
        return x

    @classmethod
    def from_model_spec(cls, spec):
        features = ConvFeatureExtractor.from_model_spec(spec['features'])
        hidden = Hidden
        output = VectorizationOutput.from_model_spec(spec['output'])
        return cls(features, hidden, output)