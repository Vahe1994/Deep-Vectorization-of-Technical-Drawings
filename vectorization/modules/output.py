import torch
import torch.nn as nn

from .base import ParameterizedModule


class Output(ParameterizedModule):
    def __init__(self, in_features=128, out_features=6, **kwargs):
        super().__init__(**kwargs)
        self.final_fc = nn.Linear(in_features, out_features)
        self.final_tanh = nn.Tanh()
        self.final_sigm = nn.Sigmoid()

    def forward(self, vector_features):
        # vector_features: [b, n, hidden_dim]
        fc = self.final_fc(vector_features)  # [b, n, output_dim]
#
        if(fc.shape[2]<=6):
            #for lines coord from 0 to 1
            coord = (self.final_tanh(fc[..., :-1]) + 1.) / 2.  # [b, n, output_dim - 1]
        else:
            coord = fc[..., :-1]  # [b, n, output_dim - 1]
        prob = self.final_sigm(fc[..., -1]).unsqueeze(-1)  # [b, n, 1]
        return torch.cat((coord, prob), dim=-1)  # [b, n, output_dim]


output_module_by_kind = {
    'output': Output,
}