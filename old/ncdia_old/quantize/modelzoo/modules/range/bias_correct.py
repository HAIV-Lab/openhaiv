"""
Bias correction
version: 0.1.0
update: 2024-01-18
"""
import torch
import torch.nn as nn
from torch import Tensor


class BiasCorrect(nn.Module):
    """ Bias correction.
        Count the expected mean of calibration data :math:`\mathbb{E}(x)`,
        and correct the bias of quantized weight :math:`\hat{W}` by
        :math:`(W-\hat{W})\mathbb{E}(x)`.

    Args:
        momentum (float): momentum of expected mean, Default: 0.1

    Returns:
        Tensor: bias

    Examples::

        >>> bc = BiasCorrect()
        >>> bc.calibrate(x)
        >>> bc(deltaW, func)

    """
    def __init__(
        self,
        momentum: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('EX', torch.tensor(0.))

    def calibrate(self, x: Tensor):
        """ calibrate bias.
        Args:
            x (Tensor): input data
        """
        self.EX = self.momentum * x.mean(dim=0, keepdim=True) + \
                  (1 - self.momentum) * self.EX
        
    def __call__(self, referX: Tensor, deltaW: Tensor, func: callable):
        """ Correct bias according to the change of weight.
        Args:
            referX (Tensor): reference input data, providing shape information
            deltaW (Tensor): change of weight, denoted as :math:`W-\hat{W}`
            func (callable): function to calculate bias,
                with signature func(x, deltaW, bias) -> Tensor
        Return:
            Tensor: bias
        """
        if self.EX.dim() == 0 and referX:
            shape = [1, *referX.shape[1:]]
            EX = self.EX.expand(*shape).to(deltaW.device)
        elif self.EX.dim() > 0:
            EX = self.EX.to(deltaW.device)
        else:
            return self.EX.to(deltaW.device)

        return func(EX, deltaW, None).mean(dim=0)
    
    def extra_repr(self) -> str:
        return f'momentum={self.momentum}'
    
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, 
        strict, missing_keys, unexpected_keys, error_msgs
    ):
        if prefix + 'EX' in state_dict:
            self.EX = state_dict[prefix + 'EX'].to(self.EX.device)
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, 
            strict, missing_keys, unexpected_keys, error_msgs)
