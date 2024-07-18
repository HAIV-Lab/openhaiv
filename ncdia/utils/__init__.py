from .registry import Registry
from .logger import Logger
from .tools import *
from .accuracy import accuracy


# manage all kinds of trainers
TRAINERS = Registry()

# manage all kinds of hooks
HOOKS = Registry()

# manage all kinds of losses
LOSSES = Registry()

# manage all kinds of algorithms
ALGORITHMS = Registry()

# mange all kinds of inc algorithms
INMETHODS = Registry()
