import inspect
import logging
from typing import TYPE_CHECKING, Any, Optional, Union

from .registry import Registry
from ncdia_new.utils.config import Config

def build_from_cfg(
    cfg: Union[dict, ConfigDict, Config],
    registry: Registry,
    default_args: Optional[Union[dict, ConfigDict, Config]] = None):
        
    if not isinstance(registry, Registry):
        raise TypeError('registry must be a mmengine.Registry object, '
                        f'but got {type(registry)}')

    if not isinstance(cfg, (dict, Config)):
        raise TypeError(
            f'cfg should be a dict, Config, but got {type(cfg)}')
    
    args = cfg.copy()
    obj_type = args.type
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.scope}::{registry.name} registry. '
            )
    elif callable(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}'
        )
    
    if inspect.isclass(obj_cls):
        obj = obj_cls(**args)
    
    return obj


    

    