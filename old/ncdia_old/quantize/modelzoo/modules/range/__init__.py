"""
Range estimator
version: 0.1.0
update: 2023-12-22
"""
from .minmax import MinMax, MAMinMax
from .mse import MSE
from .cross_entropy import CrossEntropy
from .bias_correct import BiasCorrect
from .aciq import ACIQ
from .adaround import AdaRound

from ncdia_old.quantize.utils import Register

RANGES = Register({
    'minmax': MinMax,
    'maminmax': MAMinMax,
    'mse': MSE,
    'cross_entropy': CrossEntropy,
    'bias_correct': BiasCorrect,
    'aciq': ACIQ,
    'adaround': AdaRound,
})
