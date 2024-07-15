from .registry import Registry
from .logger import Logger
from .tools import *

# manage all kinds of trainers
TRAINERS = Registry()

# manage all kinds of hooks
HOOKS = Registry()
