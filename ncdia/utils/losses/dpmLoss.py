import torch
import torch.nn as nn
import torch.nn.functional as F
from ncdia.utils import LOSSES

@LOSSES.register
class DPMLoss(nn.Module):
    '''DPM Loss 
    
    Args:
        
    '''
    def __init__(self, **kwargs):
        super(DPMLoss, self).__init__(**kwargs)

    def forward(self, input, target, factor1=0, factor2=1, factor3=0):
        """
        Args:
            input (tuple): A tuple containing:
                - logits1 (torch.Tensor): Fv @ Ft.
                - logits2 (torch.Tensor): featv @ Ft.
                - logits3 (torch.Tensor): Featvv @ visual_prototype.
            target (torch.Tensor): The target labels.
            factor1 (float): Weight for loss1.
            factor2 (float): Weight for loss2.
            factor3 (float): Weight for loss3.

        Returns:
            torch.Tensor: The total loss.
        """
        logits1, logits2, logits3 = input

        loss1 = F.cross_entropy(20*logits1, target)
        loss2 = F.cross_entropy(20*logits2, target)
        loss3 = F.cross_entropy(20*logits3, target)

        loss = factor1 * loss1 + factor2 * loss2 + factor3 * loss3

        return loss
