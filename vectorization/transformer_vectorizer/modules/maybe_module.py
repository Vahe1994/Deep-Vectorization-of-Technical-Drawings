import torch.nn as nn


class MaybeModule(nn.Module):
    def __init__(self, maybe=False, layer=None):
        super().__init__()
        self.maybe = maybe
        self.layer = layer

    def forward(self, input):
        if self.maybe:
            return self.layer.forward(input)
        return input

    def __repr__(self):
        main_str = '{}({}) {}'.format(
            self._get_name(), self.maybe, repr(self.layer))
        return main_str
