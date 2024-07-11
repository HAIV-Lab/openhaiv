import torch
import torch.nn as nn


def build_optimizer(
        name: str,
        model: nn.Module,
        param_groups: dict | None = None,
        **kwargs
    ) -> torch.optim.Optimizer:
    """Build optimizer.

    Args:
        name (str): name of optimizer
        model (nn.Module | dict): model or param_groups
        param_groups (dict | None): 
            if provided, directly optimize param_groups and abandon model
        kwargs (dict): arguments for optimizer

    Returns:
        optimizer (torch.optim.Optimizer): optimizer

    Raises:
        NotImplementedError: optimizer not supported

    Examples:
        >>> model = nn.Linear(10, 1)
        >>> optimizer = build_optimizer('adam', model)    
    """
    if param_groups is None:
        if isinstance(model, nn.Module):
            param_groups = model.parameters()
        else:
            param_groups = model

    name = name.lower()
    if name == 'adam':
        optimizer = torch.optim.Adam(param_groups, **kwargs)
    elif name == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, **kwargs)
    elif name == 'sgd':
        optimizer = torch.optim.SGD(param_groups, **kwargs)
    elif name == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_groups, **kwargs)
    else:
        raise NotImplementedError(f"Optimizer {name} not supported")

    return optimizer
