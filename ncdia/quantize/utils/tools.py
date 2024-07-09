"""
Tools
version: 0.0.2
update: 2023-12-15
"""
import torch
import random
import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MovingAverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self, max_len=100):
        self.max_len = max_len
        self.reset()
    
    def reset(self):
        self.values = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.values.extend([val] * n)
        if len(self.values) >= self.max_len:
            for _ in range(len(self.values) - self.max_len):
                self.sum -= self.values.pop(0)
                self.count -= 1       

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def dict_merge(*dicts):
    """ Merge dicts.
    Args:
        *dicts (list): dicts to merge
    Returns:
        dict: merged dict
    """
    merged = {}
    for d in dicts:
        for k, v in d.items():
            if not isinstance(v, dict):
                merged[k] = v
            else:
                merged[k] = dict_merge(merged.get(k, {}), v)
    return merged
