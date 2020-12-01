import torch.nn as nn

from vectorization.modules.base import ParameterizedModule
from vectorization.modules.maybe_module import MaybeModule


class _BasicLinear(nn.Module):
    """A n-layer-feed-forward-layer module."""

    def __init__(self, in_features, out_features, normalization='batch_norm', dropout=0.):
        super(_BasicLinear, self).__init__()

        # callable functions used for creating classes
        normalization_modules = {
            'batch_norm': nn.BatchNorm1d,
            'instance_norm': nn.InstanceNorm1d,
            'none': lambda num_features: MaybeModule(False, None)
        }

        self.layer = nn.Sequential(*(
            nn.Linear(in_features, out_features),
            normalization_modules[normalization](out_features),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout)
        ))

    def forward(self, features):
        return self.layer(features)


class LinearBlockSequence(ParameterizedModule):
    def __init__(self, layers, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.Sequential(*layers)

    @classmethod
    def from_spec(cls, spec):
        layers = [_BasicLinear(**layer_spec) for layer_spec in spec['layers']]
        return cls(layers)

    def forward(self, conv_features, max_lines):
        """
        :param conv_features: [b, h * w, c] batch of image conv features
        :param max_lines: how many lines per image to predict
        """
        out = conv_features.reshape(conv_features.shape[0], -1)  # [b, h * w * c]
        for layer in self.layers:
            out = layer(out)
        out = out.reshape(out.shape[0], max_lines, -1)
        return out


fc_module_by_kind = {
    'linear_block_seq': LinearBlockSequence,
}
