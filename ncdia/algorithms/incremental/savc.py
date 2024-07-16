import copy
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm


from ncdia.utils.cfg import Configs
from ncdia.utils.logger import Logger
from ncdia.utils import INMETHODS
from .base import BaseLearner

@INMETHODS.register()
class SAVC(BaseLearner):
    def __init__(self, cfg: Configs) -> None:
        self.args = cfg.copy()
        super().__init__(self.args)

        self._network = None


        # self._device = self.args

    
    def _init_train(self, trainloader, testloader, optimizer, scheduler):
        self._network.train()

        loss_avg = 0.0
        

    def _train(self):
        pass

    def _incremental_train(self):
        pass