import torch
from torch import nn
from ncdia.utils import LOSSES

@LOSSES.register
class RD4ADLoss(nn.Module):
    """RD4AD Loss using cosine similarity.
    
    This loss computes the cosine similarity between two sets of features
    and minimizes the dissimilarity.

    Args:
        None
    """
    def __init__(self):
        super(RD4ADLoss, self).__init__()
        self.cos_loss = nn.CosineSimilarity(dim=-1)

    def forward(self, a, b):
        """
        Forward function to compute the RD4AD loss.

        Args:
            a (list[Tensor]): List of feature tensors from one source.
            b (list[Tensor]): List of feature tensors from another source.

        Returns:
            Tensor: The computed loss value.
        """
        loss = 0
        for item in range(len(a)):
            # Flatten the tensors and compute cosine similarity
            loss += torch.mean(1 - self.cos_loss(
                a[item].view(a[item].shape[0], -1),
                b[item].view(b[item].shape[0], -1)
            ))
        return loss