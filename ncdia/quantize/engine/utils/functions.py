import torch
from torch import Tensor
import torch.nn.functional as F


def linear(input: Tensor, weight: Tensor, bias: Tensor = None):
    return F.linear(input, weight, bias)
