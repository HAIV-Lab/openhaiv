from .accuracy import *
from .meter import *

from ncdia.utils import METRICS


METRICS.register_dict({
    'base': BaseMeter,
    'average': AverageMeter,
})
