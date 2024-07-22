from .registry import Registry
from .logger import Logger
from .cfg import Configs
from .tools import *


# manage all kinds of trainers
TRAINERS = Registry()

# manage all kinds of hooks
HOOKS = Registry()

# manage all kinds of losses
LOSSES = Registry()

# manage all kinds of metrics
METRICS = Registry()

# manage all kinds of algorithms
ALGORITHMS = Registry()

# mange all kinds of inc algorithms
INMETHODS = Registry()

# manage all kinds of datasets
DATASETS = Registry()

from .losses import CrossEntropyLoss
from .metrics import accuracy
