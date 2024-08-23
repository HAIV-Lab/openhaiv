"""
Quant modules
version: 0.0.2
update: 2023-12-28
"""
from .quantizer import Quantizer
from .quantlinear import QuantLinear
from .quantconv2d import QuantConv2d
from .quantrelu import QuantReLU
from .quant_pooling import QuantMaxPool2d, QuantAdaptiveAvgPool2d

from .range import RANGES
from ncdia.quantize.utils import Register

MODULES = Register({
    'quantizer': Quantizer,
    'quantlinear': QuantLinear,
    'quantconv2d': QuantConv2d,
    'quantrelu': QuantReLU,
    'quantmaxpool2d': QuantMaxPool2d,
    'quantadaptiveavgpool2d': QuantAdaptiveAvgPool2d,
})
