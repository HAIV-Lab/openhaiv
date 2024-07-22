

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
