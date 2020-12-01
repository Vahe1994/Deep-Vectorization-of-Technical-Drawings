import torch
import torch.nn as nn

from vectorization.modules.base import ParameterizedModule
# TODO @mvkolos: parameterize the model with reasonable layer sizes


class GlobalMaxPooling(nn.Module):
    def __init__(self, dim=-1):
        super(self.__class__, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(dim=self.dim)[0]


class GlobalPooling(nn.Module):
    def __init__(self, dim=-1):
        super(self.__class__, self).__init__()
        self.dim = dim

    def forward(self, x):
        avg = x.mean(dim=self.dim)
        max = x.max(dim=self.dim)[0]
        min = x.min(dim=self.dim)[0]

        return torch.cat([min, avg, max], dim=-1)


class FullyConvolutionalNet(ParameterizedModule):
    def __init__(self, hidden_dim=128, input_channels=1, pooling='max'):
        super().__init__()
        self.hidden_dim = hidden_dim
        if pooling == 'max':
            self.pooling = nn.MaxPool2d
            self.adpooling = nn.AdaptiveMaxPool2d
        else:
            self.pooling = nn.AvgPool2d
            self.adpooling = nn.AdaptiveAvgPool2d

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=(2, 2)),
            nn.LeakyReLU(),
            self.pooling((2, 2)),

            nn.Conv2d(128, 128, kernel_size=(3, 3)),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3)),
            self.pooling((2, 2)),
            nn.Conv2d(128, 64, kernel_size=(3, 3)),
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            self.adpooling((10, 6)),  #FIXME (n_primitives, n_predicted_params)
            GlobalMaxPooling(dim=1),
            nn.Sigmoid()
        )

    def forward(self, images, n):
        img_code = self.conv(images)  # [b, c, h, w]
        return img_code  # .transpose(1, 2)  #[b, n, output_dim]

