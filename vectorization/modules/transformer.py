import torch
import torch.nn as nn

from ._transformer_modules import TransformerLayer, get_sinusoid_encoding_table
from vectorization.modules.base import ParameterizedModule


class _InternalSequentialTransformerDecoder(nn.Module):
    def __init__(self, feature_dim=128, ffn_dim=512, n_head=8, num_layers=1, **kwargs):
        """
        :param feature_dim: Wq, Wk, Wv embedding matrixes share this dimension
        :param ffn_dim: size of FC layers in TransformerLayers
        :param n_head: number of heads in TransformerLayers
        :param num_layers: number of TransformerLayers stacked together
        """
        super(_InternalSequentialTransformerDecoder, self).__init__()
        self.transformer = nn.Sequential(*(
            TransformerLayer(feature_dim, d_inner=ffn_dim, n_head=n_head)
            for _ in range(num_layers)
        ))
        self.feature_dim = feature_dim

    def forward(self, conv_features, hidden_encoding):
        # conv_features: [b, h * w, c]
        # hidden_encoding: [b, n, c]
        h_dec = hidden_encoding

        for layer in self.transformer:
            h_dec = layer(h_dec, conv_features)

        return h_dec


class TransformerBase(ParameterizedModule):
    def __init__(self, feature_dim=128, ffn_dim=512, n_head=8, num_layers=1, **kwargs):
        """
        :param feature_dim: Wq, Wk, Wv embedding matrixes share this dimension
        :param ffn_dim: size of FC layers in TransformerLayers
        :param n_head: number of heads in TransformerLayers
        :param num_layers: number of TransformerLayers stacked together
        """
        super(TransformerBase, self).__init__(**kwargs)
        self.decoder = _InternalSequentialTransformerDecoder(
            feature_dim=feature_dim,
            ffn_dim=ffn_dim,
            n_head=n_head,
            num_layers=num_layers,
        )

        self.feature_dim = feature_dim


class TransformerDecoder(TransformerBase):
    def forward(self, conv_features, max_lines):
        """
        :param conv_features: [b, c, h, w] batch of image conv features
        :param max_lines: how many lines per image to predict
        """
        sine_enc = get_sinusoid_encoding_table(max_lines, self.feature_dim, scale=1)[None]
        h_dec = torch.cat([sine_enc] * conv_features.shape[0], dim=0)   # [b, max_lines, feature_dim]
        h_dec = h_dec.to(conv_features.device)
        decoding = self.decoder(conv_features, h_dec)
        return decoding


class TransformerDiscriminator(TransformerBase):
    LINE_DIM = 6
    def __init__(self, **kwargs):
        super(TransformerDiscriminator, self).__init__(**kwargs)
        self.fc = nn.Linear(self.LINE_DIM, self.feature_dim)

    def forward(self, conv_features, predicted_lines):
        """
        :param conv_features: [b, c, h, w] batch of image conv features
        :param predicted_lines: [b, n, line_dim] batch of predicted n lines per image
        """
        h_dec = self.fc(predicted_lines)
        decoding = self.decoder(conv_features, h_dec)
        return decoding


transformer_module_by_kind = {
    'transformer_decoder': TransformerDecoder,
    'transformer_discriminator': TransformerDiscriminator,
}
