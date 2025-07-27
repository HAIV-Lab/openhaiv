import torch
import torch.nn as nn
import torch.nn.functional as F
from .sce import soft_cross_entropy
from ncdia.utils import LOSSES


@LOSSES.register
class ReweightedCrossEntropyLoss(nn.Module):
    """Reweighted Cross Entropy Loss.

    This loss applies sample-specific weights to the standard cross-entropy loss.

    Args:
        None
    """

    def __init__(self):
        super(ReweightedCrossEntropyLoss, self).__init__()

    def forward(self, logits, labels, sample_weights):
        """
        Args:
            logits (Tensor): Model predictions of shape [batch_size, num_classes].
            labels (Tensor): Ground truth labels of shape [batch_size].
            sample_weights (Tensor): Weights for each sample of shape [batch_size].

        Returns:
            Tensor: The computed loss value.
        """
        losses = F.cross_entropy(logits, labels, reduction="none")
        return (losses * sample_weights.type_as(losses)).mean()


@LOSSES.register
class ReweightedSoftCrossEntropyLoss(nn.Module):
    """Reweighted Soft Cross Entropy Loss.

    This loss applies sample-specific weights to the soft cross-entropy loss.

    Args:
        None
    """

    def __init__(self):
        super(ReweightedSoftCrossEntropyLoss, self).__init__()

    def forward(self, logits, soft_labels, sample_weights):
        """
        Args:
            logits (Tensor): Model predictions of shape [batch_size, num_classes].
            soft_labels (Tensor): Soft labels of shape [batch_size, num_classes].
            sample_weights (Tensor): Weights for each sample of shape [batch_size].

        Returns:
            Tensor: The computed loss value.
        """
        losses = soft_cross_entropy(logits, soft_labels, reduce=False)
        return torch.mean(losses * sample_weights.type_as(losses))
