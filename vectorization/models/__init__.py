import json

from .generic import GenericVectorizationNet
from .fully_conv_net import FullyConvolutionalNet
from .lstm import LSTMTagger, LSTMTagger_normal, LSTMTagger_attent
from vectorization.modules.base import load_with_spec


MODEL_BY_NAME = {
    'FullyConvolutionalNet': FullyConvolutionalNet,
    'LSTMTagger': LSTMTagger,
    'LSTMTagger_normal': LSTMTagger_normal,
    'LSTMTagger_attent': LSTMTagger_attent,
    'GenericVectorizationNet': GenericVectorizationNet,
}


def load_model(spec_filename, checkpoint=None):
    model_spec = None
    if None is not spec_filename:
        with open(spec_filename) as model_spec_file:
            model_spec = json.load(model_spec_file)
    model = load_with_spec(model_spec, MODEL_BY_NAME)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model


__all__ = [
    'load_model'
]
