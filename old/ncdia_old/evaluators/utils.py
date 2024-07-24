from ncdia_old.utils import Config

from .attr_evaluator import AttrEvaluator
from .base_evaluator import BaseEvaluator
from .ece_evaluator import ECEEvaluator
from .ood_evaluator import OODEvaluator
from .savc_attr_evaluator import SAVCattEvaluator
from .savc_evaluator import SAVCEvaluator
# from osr_evaluator import OSREvaluator

from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

import ncdia_old.utils.comm as comm


def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'attr': AttrEvaluator,
        'ood': OODEvaluator,
        'ece': ECEEvaluator,
        'savc_attr': SAVCattEvaluator,
        'savc': SAVCEvaluator,
    }
    return evaluators[config.evaluator.name](config)

