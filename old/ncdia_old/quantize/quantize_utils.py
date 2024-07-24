from .modelzoo import reconstruct, Quantizer, RANGES
from .utils import Configs
from .optim import build_optimizer, build_lr_scheduler


quant_cfg = Configs({
    "name": "adaround",
    "quant": {
        "default": {
            "weight": {
                "n_bits": 8,
                "symmetric": True,
                "signed": True,
                "granularity": 'channel',
                "range": {
                    "name": 'mse',
                    "maxshrink": 0.8,
                    "grid": 100,
                },
                "adaround": {
                    "apply": True,
                },
            },
            "activation": {
                "n_bits": 32,
                "range": {
                    "name": "minmax",
                },
            },
            "bn_folding": True,
            "bias_correct": {
                "momentum": 0.1,
            },
        },
    },
    "optimizer": {
        "name": "adam",
        "lr": 1e-3,
    },
    "train": {
        # "max_epoch": 1,
        "max_epoch": 10,
    },
})


def quant_train(model, train=True, quantized=False):
    if train:
        model.train()
    else:
        model.eval()

    for module in model.modules():
        if hasattr(module, 'calibrating'):
            module.calibrating = train
        if isinstance(module, Quantizer):
            module.quant(quantized)

def quant_eval(model, quantized=False):
    quant_train(model, False, quantized)

def adaround_params(model):
    adaround_modules = []
    param_groups = []

    for module in model.modules():
        if isinstance(module, RANGES['adaround']):
            module.requires_grad = True
            adaround_modules.append(module)
            for param in module.parameters():
                param_groups.append(param)
        else:
            module.requires_grad = False

    return adaround_modules, param_groups

def register_forward_hook(model, buffer):
    def hook(module, _, output):
        name: str = module.__class__.__name__
        if not name.startswith('Quant') or name == 'Quantizer':
            return
        else:
            buffer.append(output)
    
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook))

    return hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()

def get_beta(current_iter, total_iter, start_val=20, end_val=2, warmup=0.2):
    if current_iter / total_iter < warmup:
        return start_val
    else:
        return start_val + (end_val - start_val) * (current_iter / total_iter - warmup) / (1 - warmup)
