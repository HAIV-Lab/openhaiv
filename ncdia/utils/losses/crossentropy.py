import torch.nn as nn
from ncdia.utils import LOSSES


@LOSSES.register
class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss with label smoothing.

    Args:
        smoothing (float): The smoothing factor. Default: 0.0.
    """

    def __init__(self, smoothing: float = 0.0, **kwargs):
        super(CrossEntropyLoss, self).__init__(**kwargs)
        self.smoothing = smoothing

    def forward(self, input, target):
        """Forward function.

        Args:
            input (Tensor): The input tensor.
            target (Tensor): The target tensor.
        """
        if self.smoothing > 0:
            assert target.dim() > 1
            n_class = input.size(1)
            target = target.clone()
            target[target == 1] = 1 - self.smoothing
            target[target == 0] = self.smoothing / (n_class - 1)
        return super(CrossEntropyLoss, self).forward(input, target)
