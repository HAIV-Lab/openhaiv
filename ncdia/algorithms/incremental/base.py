import copy
import torch
import torch.nn as nn

import numpy as np

from ncdia.utils.cfg import Configs
from ncdia.utils.logger import Logger
from ncdia.utils import INMETHODS

@INMETHODS.register()
class BaseLearner(object):
    """ Base Class for incremental learning all the inc methods should follow this class
        Args:
        cfg: Configs 
    """
    def __init__(self, cfg: Configs) -> None:
        self.args = cfg.copy()
        self._cur_taks = -1
        self._known_class = 0
        self._total_class = 0
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        # self._device = self.args

    
    def _train(self):
        pass

    def _incremental_train(self):
        pass