import torch.nn as nn

from vectorization.modules.base import ParameterizedModule, load_with_spec
from vectorization.modules import module_by_kind
from vectorization.modules.maybe_module import MaybeModule


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn=True, conv_class=nn.Conv1d):
        super().__init__()
        bn_class = {nn.Conv1d: nn.BatchNorm1d, nn.Conv2d: nn.BatchNorm2d}[conv_class]
        modules = (conv_class(in_channels, out_channels, kernel_size),
                   MaybeModule(bn, bn_class(out_channels)),
                   nn.LeakyReLU(inplace=True))
        self.block = nn.Sequential(*modules)


class ConvAdapter(ConvBlock):
    def __init__(self, in_channels, out_channels, bn=True):
        super().__init__(in_channels, out_channels, kernel_size=(1, 1), bn=bn, conv_class=nn.Conv2d)

    def forward(self, conv_features):
        return self.block(conv_features)


class TransformerAdapter(ConvBlock):
    def __init__(self, in_channels, out_channels, bn=True):
        super().__init__(in_channels, out_channels, kernel_size=1, bn=bn)

    def forward(self, decoded):
        decoded = decoded.transpose(1, 2)
        decoded = self.block(decoded)
        decoded = decoded.transpose(1, 2)
        return decoded


class GenericVectorizationNet(ParameterizedModule):
    def __init__(self, features, hidden, output, fc_from_conv, fc_from_hidden):
        super().__init__()
        self.features = features
        self.hidden = hidden
        self.output = output
        self.fc_from_conv = fc_from_conv
        self.fc_from_hidden = fc_from_hidden

    def forward(self, images, n):
        x = self.features(images)
        x = self.fc_from_conv(x)
        x = x.reshape([x.shape[0], x.shape[1], -1]) \
            .transpose(1, 2)  # [b, h * w, c]
        x = self.hidden(x, n)
        x = self.fc_from_hidden(x)
        x = self.output(x)
        return x

    @classmethod
    def from_spec(cls, spec):
        features = load_with_spec(spec['conv'], module_by_kind)
        hidden = load_with_spec(spec['hidden'], module_by_kind)
        output = load_with_spec(spec['output'], module_by_kind)

        use_fc_from_conv = spec.get('use_fc_from_conv', False)
        # conv1x1_conv2hid = nn.Conv1d(spec['conv']['convmap_channels'], spec['hidden']['feature_dim'], (1, 1))
        if use_fc_from_conv:
            conv_adapter = ConvAdapter(spec['conv']['convmap_channels'],
                                       spec['hidden']['feature_dim'],
                                       bn=spec['conv']['bn'])
        else:
            conv_adapter = None
        fc_from_conv = MaybeModule(maybe=use_fc_from_conv, layer=conv_adapter)

        # conv1x1_hid2out = nn.Conv1d(spec['hidden']['feature_dim'], spec['output']['hidden_dim'], 1)
        use_fc_from_hidden = spec.get('use_fc_from_hidden', False)
        if use_fc_from_hidden:
            transformer_adapter = TransformerAdapter(spec['hidden']['feature_dim'], spec['output']['in_features'], bn=False)
        else:
            transformer_adapter = None
        fc_from_hidden = MaybeModule(maybe=use_fc_from_hidden, layer=transformer_adapter)

        return cls(features, hidden, output, fc_from_conv, fc_from_hidden)


__all__ = [
    "GenericVectorizationNet"
]
