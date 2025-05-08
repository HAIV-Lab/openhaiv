import torch
import torch.nn as nn


def build_optimizer(
        type: str,
        model: nn.Module,
        param_groups: dict | None = None,
        **kwargs
    ) -> torch.optim.Optimizer:
    """Build optimizer.

    Args:
        type (str): type of optimizer
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

    # print(f"model.parameters: {model.parameters}")
    type = type.lower()
    if type == 'adam':
        optimizer = torch.optim.Adam(param_groups, **kwargs)
    elif type == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, **kwargs)
    elif type == 'sgd':
        optimizer = torch.optim.SGD(param_groups, **kwargs)
    elif type == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_groups, **kwargs)
    else:
        raise NotImplementedError(f"Optimizer {type} not supported")

    return optimizer
