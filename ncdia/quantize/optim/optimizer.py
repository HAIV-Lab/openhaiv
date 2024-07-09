"""
Optimizer.
version: 0.0.1
update: 2023-12-22
"""
import torch
import torch.nn as nn


def build_optimizer(model, cfg, param_groups=None):
    """Build optimizer.

    Args:
        model (nn.Module or iterable): model
        cfg (Configs): configs
        param_groups (iterable, optional): If provided, directly optimize param_groups and abandon model
    """
    # assert 'optimizer' in cfg.__dict__, "optimizer not found in configs"
    optim = cfg.optimizer

    if param_groups is None:
        if isinstance(model, nn.Module):
            param_groups = model.parameters()
        else:
            param_groups = model

    if optim.name == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=optim.lr or 1e-3,
            betas=(optim.beta1 or 0.9, optim.beta2 or 0.999),
            weight_decay=optim.weight_decay or 0,
        )

    elif optim.name == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=optim.lr or 1e-3,
            betas=(optim.beta1 or 0.9, optim.beta2 or 0.999),
            weight_decay=optim.weight_decay or 1e-2,
        )

    elif optim.name == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=optim.lr or 1e-3,
            momentum=optim.momentum or 0,
            weight_decay=optim.weight_decay or 0,
            nesterov=optim.nesterov or False,
        )

    elif optim.name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=optim.lr or 1e-2,
            alpha=optim.alpha or 0.99,
            weight_decay=optim.weight_decay or 0,
            momentum=optim.momentum or 0,
        )

    else:
        raise NotImplementedError(f"Optimizer {optim.name} not supported")

    return optimizer
