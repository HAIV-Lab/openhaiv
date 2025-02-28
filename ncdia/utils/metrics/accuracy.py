import torch

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k-top predictions for the specified values of k.

    Args:
        output (torch.Tensor): model output, shape (batch_size, num_classes)
        target (torch.Tensor): target labels, shape (batch_size)
        topk (tuple): top-k values, default is (1,)

    Returns:
        acc (list): accuracy values for each k in topk
    """
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

# def per_class_accuracy(output, target, topk=(1,)):
#     """ Compute per class accuracy over the k-top predictions for the specified values of k 
    
#     Args:
#         output (torch.Tensor): model output, shape (batch_size, num_classes)
#         target (torch.Tensor): target labels, shape (batch_size)
#         topk (tuple): top-k values, default is (1,)

#     Returns:
#         per_class_acc (dict): accuracy values for each class
#     """
#     maxk = max(topk)
#     batch_size = target.size(0)
#     unique_class = torch.unique(target)
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     # Compute class-wise accuracy
#     per_class_acc = {c.item(): 0.0 for c in unique_class}

#     for c in unique_class:
#         class_mask = (target == c)
#         class_count = class_mask.sum().item()

#         if class_count > 0:  # Avoid division by zero
#             correct_k = correct[:maxk][class_mask].sum().item()
#             per_class_acc[c.item()] = (correct_k / class_count) * 100.0

#     return per_class_acc


def per_class_accuracy(output, target, topk=(1, )):
    """ Compute per class accuracy over the k-top predictions for the specified values of k 
    
    Args:
        output (torch.Tensor): model output, shape (batch_size, num_classes)
        target (torch.Tensor): target labels, shape (batch_size)
        topk (tuple): top-k values, default is (1,)

    Returns:
        per_class_acc (list): accuracy values for each k in topk
    """
    maxk = max(topk)
    batch_size = target.size(0)
    unique_class = torch.unique(target)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        
        # Compute class-wise accuracy
        per_class_acc = []
        for c in unique_class:
            class_mask = (target == c)
            class_mask = class_mask.unsqueeze(0)
            class_correct = (pred[:k][class_mask].unique().size(0) / 
                            class_mask.sum().item() * 100)
            per_class_acc.append(round(class_correct, 2))
    return per_class_acc
