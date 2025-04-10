import torch
import torch.nn as nn
import torch.nn.functional as F
from ncdia.utils import LOSSES

def entropy_select_topk(p, top_k, label, num_of_local_feature):
    """
    Extract non-Top-K regions and calculate entropy.
    Args:
        p (torch.Tensor): The input tensor.
        top_k (int): The number of top K classes to select.
        label (torch.Tensor): The label tensor.
        num_of_local_feature (int): The number of local features.
    Returns:
        torch.Tensor: The entropy of the selected regions.
    """
    label_repeat = label.repeat_interleave(num_of_local_feature)
    p = F.softmax(p, dim=-1)
    pred_topk = torch.topk(p, k=top_k, dim=1)[1]
    contains_label = pred_topk.eq(torch.tensor(label_repeat).unsqueeze(1)).any(dim=1)
    selected_p = p[~contains_label]

    if selected_p.shape[0] == 0:
        return torch.tensor([0]).cuda()
    return -torch.mean(torch.sum(selected_p * torch.log(selected_p+1e-5), 1))

@LOSSES.register
class LoCoOpLoss(nn.Module):
    '''LoCoOp Loss = CrossEntropyLoss + Local_OOD_Entropy
    
    Args:
        num_classes (int): Number of classes.
        num_local_ood (int): Number of local OOD classes.
        topk (int): . Default: 200.
        lambda (float): Weight for the local OOD entropy loss. Default: 0.25.
    '''
