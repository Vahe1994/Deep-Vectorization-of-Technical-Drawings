from abc import ABC

import torch.nn


class ParameterizedModule(torch.nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def from_spec(cls, spec):
        return cls(**spec)


def load_with_spec(spec, module_dict=None):
    spec_kind = spec['kind']
    if spec_kind.startswith('torch.nn.'):
        # loading default torch modules
        torch_nn, classname = spec_kind.rsplit('.', maxsplit=1)
        torch_class = getattr(torch.nn, classname)
        torch_class_params = {key:value for key, value in spec.items() if key != 'kind'}
        return torch_class(**torch_class_params)
    else:
        assert spec_kind in module_dict, 'unknown kind of module: "{}"'.format(spec_kind)
        spec_cls = module_dict[spec_kind]
        if set(spec.keys()) == {spec_kind}:  # no parameterization specified
            return spec_cls()  # instantiate default parameterization
        else:
            return spec_cls.from_spec(spec)

