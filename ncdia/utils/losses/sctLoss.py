import torch
import torch.nn as nn
import torch.nn.functional as F
from ncdia.utils import LOSSES

def entropy_select_topk(p, top_k, label, num_of_local_feature, nat_probs):
    """
    Extract non-Top-K regions and calculate entropy.
    Args:
        p (torch.Tensor): The input tensor.
        top_k (int): The number of top K classes to select.
        label (torch.Tensor): The label tensor.
        num_of_local_feature (int): The number of local features.
        nat_probs (torch.Tensor): The natural probabilities tensor.
    Returns:
        torch.Tensor: The entropy of the selected regions.
    """
    true_probs = torch.gather(nat_probs, 1, (label.unsqueeze(1)).long()).squeeze()
    true_probs_repeat = true_probs.repeat_interleave(num_of_local_feature)

    label_repeat = label.repeat_interleave(num_of_local_feature)
    p = F.softmax(p, dim=-1)
    pred_topk = torch.topk(p, k=top_k, dim=1)[1]
    contains_label = pred_topk.eq(torch.tensor(label_repeat).unsqueeze(1)).any(dim=1)
    selected_p = p[~contains_label]
    selected_true_probs = true_probs_repeat[~contains_label]

    if selected_p.shape[0] == 0:
        return torch.tensor([0]).cuda()

    return -torch.mean(torch.sum(selected_p * torch.log(selected_p+1e-5), 1) * (0.0000001 + selected_true_probs))  

@LOSSES.register
class SCTLoss(nn.Module):
    '''LoCoOp Loss = CrossEntropyLoss + Local_OOD_Entropy
    
    Args:
        num_classes (int): Number of classes.
        num_local_ood (int): Number of local OOD classes.
        topk (int): The number of top K classes to select.
        lambda (float): Weight for the local OOD entropy loss. .
    '''
    def __init__(self, **kwargs):
        super(SCTLoss, self).__init__(**kwargs)

    def forward(self, input, target, topk=20, lambda_en=0.25):
        """
        Args:
            input (tuple): A tuple containing:
                - logits_global (torch.Tensor): The global logits.
                - logits_local (torch.Tensor): The local logits.
            target (torch.Tensor): The target labels.
            topk (int): The number of top K classes to select.
            num_of_local_feature (int): The number of local features.
        Returns:
            torch.Tensor: The total loss.
        """
        logits_global, logits_local = input

        # calculate CoOp loss
        label_onehot = F.one_hot(target, num_classes=94)
        probs = F.softmax(logits_global, dim=1)
        true_probs = torch.gather(probs, 1, (target.unsqueeze(1)).long()).squeeze()
        loss = -torch.sum(label_onehot * F.log_softmax(logits_global, dim=1), dim=1) * (1.0000001 - true_probs)
        loss_id = loss.mean()

        # calculate OOD regularization loss
        batch_size, num_of_local_feature = logits_local.shape[0], logits_local.shape[1]
        logits_local = logits_local.view(batch_size * num_of_local_feature, -1)     
        loss_en = - entropy_select_topk(logits_local, topk, target, num_of_local_feature, probs)

        # calculate total loss for sct
        loss = loss_id + lambda_en * loss_en

        return loss
