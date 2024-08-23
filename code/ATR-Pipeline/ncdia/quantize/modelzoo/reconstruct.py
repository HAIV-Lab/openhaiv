"""
Model Reconstructor
version: 0.1.2
update: 2024-01-16
"""
import re
import torch
import torch.nn as nn
from collections import OrderedDict

from ncdia.quantize.utils import Configs, dict_merge
from .modules import MODULES


def reconstruct(model: nn.Module, cfg_quant: Configs, root: str = ''):
    """ Reconstruct model using quant modules.
    Args:
        model (nn.Module): model to reconstruct
        cfg_quant (Configs): configs for quant module
        root (str): root name of the model
    Returns:
        nn.Module: reconstructed model
    """
    module_dict = OrderedDict(model.named_children())
    names = list(module_dict.keys())
    modules = list(module_dict.values())
    skip_modules = []

    for idx, (name, module) in enumerate(module_dict.items()):
        if name in skip_modules:
            model.__setattr__(name, nn.Identity())
            continue

        name, module, skip = \
            choose_modules(modules[idx:idx+2], names[idx:idx+2], cfg_quant, root)
        model.__setattr__(name, module)

        if skip:
            skip_modules.append(skip)
    
    return model


def parse_parameters(cfg_module: Configs, default: dict, name: str = '') -> dict:
    """ Parse parameters for quant module.
    Args:
        cfg_module (Configs): Configs for quant module
        default (dict): default parameters
        name (str): subset name, if specified, 
            only parameters with key matching name will be parsed
    Returns:
        dict: parsed parameters
    """
    assert isinstance(default, dict)
    if isinstance(cfg_module, Configs):
        params = cfg_module.cfg
        if name:
            params = [v for k, v in params.items() if re.match(k, name)]
            params = dict_merge(*params)
    else:
        params = cfg_module

    if not isinstance(params, dict):
        return default
    else:
        if 'weight' in params:
            params['w_setting'] = params.pop('weight')
        if 'activation' in params:
            params['a_setting'] = params.pop('activation')

        return dict_merge(default, params)
    

def get_params(cfg_quant: Configs, default: dict, names: list = []):
    """ Get parameters for quant module.
    Args:
        cfg_quant (Configs): Configs for quant module
        default (dict): default parameters
        names (list): list of subset name, if specified, 
            only parameters with key matching name will be parsed
    Returns:
        dict: parsed parameters
    """
    if not isinstance(names, list):
        names = [names]

    params = parse_parameters(cfg_quant.default, default)
    for n in names:
        params = parse_parameters(cfg_quant, params, n)
        
    return params


def choose_modules(modules: list, names: list, cfg_quant: Configs, root: str):
    """ Choose quant modules according to module type.
    Args:
        modules (list): list of modules
        names (list): list of module names
        cfg_quant (Configs): configs for quant module
        root (str): root name of the model
    Returns:
        tuple: (module name, quant module, skipped module name)
    """
    root = '/'.join([root, names[0]])

    if isinstance(modules[0], nn.Conv2d):
        if len(modules) > 1 and isinstance(modules[1], nn.BatchNorm2d) \
        and cfg_quant.default.bn_folding:
            params = get_params(cfg_quant, modules[0].__dict__, ['nn_conv2d_bn2d', root])
            return names[0], conv2d_bn2d(params, modules[1]), names[1]
        
        params = get_params(cfg_quant, modules[0].__dict__, ['nn_conv2d', root])
        return names[0], conv2d(params), ''
        
    # elif isinstance(modules[0], nn.Linear):
    #     params = get_params(cfg_quant, modules[0].__dict__, ['nn_linear', root])
    #     return names[0], linear(params), ''
    
    # elif isinstance(modules[0], nn.MaxPool2d):
    #     params = get_params(cfg_quant, modules[0].__dict__, ['nn_maxpool2d', root])
    #     return names[0], maxpool2d(params), ''
    
    # elif isinstance(modules[0], nn.AdaptiveAvgPool2d):
    #     params = get_params(cfg_quant, modules[0].__dict__, ['nn_adaptiveavgpool2d', root])
    #     return names[0], adaptiveavgpool2d(params), ''
    
    else:
        return names[0], reconstruct(modules[0], cfg_quant, root), ''

def conv2d(params: dict):
    params['bn_folding'] = False
    return MODULES['quantconv2d'](**params)

def conv2d_bn2d(params: dict, nn_bn2d: nn.BatchNorm2d):
    assert params['bn_folding']
    if not isinstance(params['bn_folding'], dict):
        params['bn_folding'] = {}

    params['bn_folding'].update({
        'running_mean': nn_bn2d.running_mean.detach().clone(),
        'running_var': nn_bn2d.running_var.detach().clone(),
        'weight': nn_bn2d.weight.detach().clone(),
        'bias': nn_bn2d.bias.detach().clone(),
        'eps': torch.tensor(nn_bn2d.eps).to(nn_bn2d.weight.device)})
    return MODULES['quantconv2d'](**params)

def linear(param: dict):
    return MODULES['quantlinear'](**param)

def maxpool2d(params: dict):
    return MODULES['quantmaxpool2d'](**params)

def adaptiveavgpool2d(params: dict):
    return MODULES['quantadaptiveavgpool2d'](**params)

def relu(params: dict):
    return MODULES['quantrelu'](**params)
