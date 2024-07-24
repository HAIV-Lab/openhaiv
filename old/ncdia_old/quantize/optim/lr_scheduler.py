"""
Learning rate scheduler.
version: 0.0.1
update: 2023-12-22
"""
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupLR(_LRScheduler):
    """Cosine learning rate scheduler with warmup epochs.

    Args:
        optimizer (torch.optim.Optimizer): optimizer
        T_max (int): maximum number of iterations
        eta_min (float): minimum learning rate
        warmup_epochs (int): number of warmup epochs
        warmup_type (str): warmup type, 'constant' or 'linear'
        warmup_lr (float): warmup learning rate
        last_epoch (int): last epoch
    """

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min=0,
        warmup_epochs=0,
        warmup_type='constant',
        warmup_lr=None,
        last_epoch=-1,
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        self.warmup_lr = warmup_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_type == 'constant':
                if self.warmup_lr is None:
                    return self.base_lrs
                else:
                    return [self.warmup_lr for _ in self.base_lrs]
            elif self.warmup_type == 'linear':
                return [
                    base_lr * self.last_epoch / self.warmup_epochs
                    for base_lr in self.base_lrs
                ]
            else:
                raise ValueError(f'Invalid warmup type: {self.warmup_type}')
        else:
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs))) / 2
                for base_lr in self.base_lrs
            ]


class LinearWarmupLR(_LRScheduler):
    """Linear learning rate scheduler with warmup epochs.

    Args:
        optimizer (torch.optim.Optimizer): optimizer
        T_max (int): maximum number of iterations
        eta_min (float): minimum learning rate
        warmup_epochs (int): number of warmup epochs
        warmup_type (str): warmup type, 'constant' or 'linear'
        warmup_lr (float): warmup learning rate
        last_epoch (int): last epoch
    """

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min=0,
        warmup_epochs=0,
        warmup_type='constant',
        warmup_lr=None,
        last_epoch=-1,
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        self.warmup_lr = warmup_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_type == 'constant':
                if self.warmup_lr is None:
                    return self.base_lrs
                else:
                    return [self.warmup_lr for _ in self.base_lrs]
            elif self.warmup_type == 'linear':
                return [
                    base_lr * self.last_epoch / self.warmup_epochs
                    for base_lr in self.base_lrs
                ]
            else:
                raise ValueError(f'Invalid warmup type: {self.warmup_type}')
        else:
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (self.T_max - self.last_epoch) / (self.T_max - self.warmup_epochs)
                for base_lr in self.base_lrs
            ]


class ConstantLR(_LRScheduler):
    """Constant learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): optimizer
        last_epoch (int): last epoch
    """

    def __init__(
        self,
        optimizer,
        last_epoch=-1,
    ):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.base_lrs


def build_lr_scheduler(optimizer, cfg):
    """Build lr scheduler.

    Args:
        optimizer (torch.optim.Optimizer): optimizer
        cfg (Configs): configs
    """
    assert 'lr_scheduler' in cfg.__dict__, "lr_scheduler not found in configs"
    sched = cfg.lr_scheduler

    if sched.name == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            sched.step_size,
            sched.gamma or 0.1,
            last_epoch=sched.last_epoch or -1,
        )

    elif sched.name == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            sched.milestones,
            sched.gamma or 0.1,
            last_epoch=sched.last_epoch or -1,
        )

    elif sched.name == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            sched.gamma,
            last_epoch=sched.last_epoch or -1,
        )

    elif sched.name == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            sched.T_max or cfg.train.max_epoch,
            sched.eta_min or 0,
            last_epoch=sched.last_epoch or -1,
        )

    elif sched.name == 'cosine_warmup':
        lr_scheduler = CosineWarmupLR(
            optimizer,
            sched.T_max or cfg.train.max_epoch,
            sched.eta_min or 0,
            sched.warmup_epochs or 0,
            sched.warmup_type or 'constant',
            sched.warmup_lr,
            last_epoch=sched.last_epoch or -1,
        )

    elif sched.name == 'linear_warmup':
        lr_scheduler = LinearWarmupLR(
            optimizer,
            sched.T_max or cfg.train.max_epoch,
            sched.eta_min or 0,
            sched.warmup_epochs or 0,
            sched.warmup_type or 'constant',
            sched.warmup_lr,
            last_epoch=sched.last_epoch or -1,
        )

    elif sched.name == 'constant':
        lr_scheduler = ConstantLR(
            optimizer,
            last_epoch=sched.last_epoch or -1,
        )

    else:
        raise NotImplementedError(f"LR scheduler {sched.name} not supported")

    return lr_scheduler


if __name__ == '__main__':
    weight = torch.randn((2, 2), requires_grad=True)
    weight.grad = torch.ones((2, 2))

    optimizer = torch.optim.SGD([weight], lr=1)
    scheduler = CosineWarmupLR(optimizer, 15, 0, 5)
    # scheduler = ConstantLR(optimizer)

    for i in range(15):
        optimizer.zero_grad()
        loss = (weight ** 2).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"epoch: {i}, lr: {optimizer.param_groups[0]['lr']}, loss: {loss.item()}")
