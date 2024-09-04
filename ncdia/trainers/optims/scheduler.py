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

    Examples:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = CosineWarmupLR(optimizer, T_max=100, eta_min=0, warmup_epochs=10)
    """
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            T_max: int,
            eta_min: float = 0,
            warmup_epochs: int = 0,
            warmup_type: str = 'constant',
            warmup_lr: float | None = None,
            last_epoch: int = -1,
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

    Examples:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = LinearWarmupLR(optimizer, T_max=100, eta_min=0, warmup_epochs=10)
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            T_max: int,
            eta_min: float = 0,
            warmup_epochs: int = 0,
            warmup_type: str = 'constant',
            warmup_lr: float | None = None,
            last_epoch: int = -1,
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

    Examples:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = ConstantLR(optimizer)
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
    ):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.base_lrs


def build_scheduler(
        type: str,
        optimizer: torch.optim.Optimizer,
        **kwargs
    ) -> _LRScheduler:
    """Build learning rate scheduler.

    Args:
        type (str): type of scheduler
        optimizer (torch.optim.Optimizer): optimizer
        kwargs (dict): arguments for scheduler

    Returns:
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler

    Raises:
        NotImplementedError: scheduler not supported

    Examples:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = build_scheduler('cosine', optimizer)
    """
    type = type.lower()
    if type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif type == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif type == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif type == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif type == 'cosine_warmup':
        lr_scheduler = CosineWarmupLR(optimizer, **kwargs)
    elif type == 'linear_warmup':
        lr_scheduler = LinearWarmupLR(optimizer, **kwargs)
    elif type == 'constant':
        lr_scheduler = ConstantLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(f"LR scheduler {type} not supported")

    return lr_scheduler
