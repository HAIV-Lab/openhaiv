import copy
import logging
import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ncdia.utils import MODELS, Configs


@MODELS.register
class ODINNet(nn.Module):
    """Net for Decoupling Maxlogit.

    Args:
        network (Configs): Network configuration.
    
    """
    def __init__(
        self,
        network: Configs,
        checkpoint: str = None,
        temperature: float = 1000.0,
        noise: float = 0.001,
        input_std: list = [0.5, 0.5, 0.5],
        **kwargs
    ) -> None:
        super().__init__()
        self.args = network.cfg
        self.network = MODELS.build(copy.deepcopy(self.args))
        self.temperature = temperature
        self.noise = noise
        self.input_std = input_std
        
        # Load the state_dict
        if checkpoint:
            state_dict = torch.load(checkpoint)
            self.network_C.load_state_dict(state_dict)
        self.out_features = None

    def forward(self, x):
        x.requires_grad = True
        output = self.network(x)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        criterion = nn.CrossEntropyLoss()

        labels = output.detach().argmax(axis=1)

        # Using temperature scaling
        output = output / self.temperature

        loss = criterion(output, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2

        # Scaling values taken from original code
        gradient[:, 0] = (gradient[:, 0]) / self.input_std[0]
        gradient[:, 1] = (gradient[:, 1]) / self.input_std[1]
        gradient[:, 2] = (gradient[:, 2]) / self.input_std[2]

        # Adding small perturbations to images
        tempInputs = torch.add(x.detach(), gradient, alpha=-self.noise)
        output = self.network(tempInputs)
        output = output / self.temperature

        # Calculating the confidence after adding perturbations
        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

        # conf, pred = nnOutput.max(dim=1)
        return nnOutput
    
    def get_features(self):
        return self.network.get_features()