"""
Adaround
version: 0.0.1
update: 2023-12-22
"""
import torch
import torch.nn as nn
from torch import Tensor


class AdaRound(nn.Module):
    """ AdaRound: Adaptive Rounding for PTQ

    Up or Down? Adaptive Rounding for Post-Training Quantization
    https://arxiv.org/abs/2004.10568

    Args:
        gamma (float): lower bound of V, default -0.1
        zeta (float): upper bound of V, default 1.1

    Returns:
        Tensor: rounded tensor

    Examples::
    
        >>> ar = AdaRound()
        >>> ar(x)

    """
    def __init__(
        self,
        gamma: float = -0.1,
        zeta: float = 1.1,
        **kwargs
    ):
        super().__init__()
        self.gamma = gamma
        self.zeta = zeta

        self.V = nn.Parameter(torch.zeros(1))
        self.ada_init = False
    
    @property
    def recV(self):
        """ rectified sigmoid,
            denoted as :math:`clip(\sigmoid(V)(\zeta-\lambda)+\lambda, 0, 1)`
        """
        return torch.clamp(
            torch.sigmoid(self.V) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_V(self, x: Tensor):
        """ Initialize V by :math:`-\log(\frac{\zeta-\gamma}{recV-\gamma}-1)`
        Args:
            x (Tensor): input tensor
        """
        recV = x - torch.floor(x)
        self.V.data = \
            -torch.log((self.zeta - self.gamma) / (recV - self.gamma) - 1)
    
    def regularization(self, beta, reduction='mean'):
        """ Regularization loss of V
        Args:
            beta (float): hyper-parameter of regularization loss
            reduction (str): reduction method, default 'mean',
                can be 'mean', 'sum' or 'none'
        Returns:
            Tensor: regularization loss caculated by 
                :math:`\sum(1-|2recV-1|^{\beta})`
        """
        regular = (1 - (2 * self.recV - 1).abs().pow(beta))
        if reduction == 'mean':
            return regular.mean()
        elif reduction == 'sum':
            return regular.sum()
        else:
            return regular

    def forward(self, x: Tensor):
        """ Round tensor x to integer
        Args:
            x (Tensor): input tensor, 
                which is divided by scale and subtracted by zero
        Returns:
            Tensor: rounded tensor
        """
        if not self.ada_init:
            self.init_V(x)
            self.ada_init = True

        x_floor = torch.floor(x)
        x_ada = x_floor + self.recV
        
        # gradient propagation
        x_ada = (x_ada.round() - x_ada).detach() + x_ada

        return x_ada
    
    def extra_repr(self) -> str:
        return f'gamma={self.gamma}, zeta={self.zeta}'
    
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, 
        strict, missing_keys, unexpected_keys, error_msgs
    ):
        self.V.data = state_dict[prefix + 'V'].to(self.V.device)
        self.ada_init = True
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, 
            strict, missing_keys, unexpected_keys, error_msgs)
