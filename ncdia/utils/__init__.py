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

# manage all kinds of preTraining algorithms
ALGORITHMS = Registry()

# manage all kinds of datasets
DATASETS = Registry()

# manage all kinds of inc network
MODELS = Registry()
