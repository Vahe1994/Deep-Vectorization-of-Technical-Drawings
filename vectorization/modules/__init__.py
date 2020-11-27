from .conv_modules import ResnetConv, VggConv, conv_module_by_kind
from .transformer import TransformerDecoder, TransformerDiscriminator, transformer_module_by_kind
from .output import output_module_by_kind
from .fully_connected import fc_module_by_kind


hidden_module_by_kind = {   # not Python 2.7 compatible!
    **transformer_module_by_kind,
    **fc_module_by_kind,
}

module_by_kind = {  # not Python 2.7 compatible!
    **conv_module_by_kind,
    **transformer_module_by_kind,
    **output_module_by_kind,
    **fc_module_by_kind,
}
