import os
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F

from .quantize_utils import *


def quantize_train_savc(quant_cfg, model, transform, trainloader):
    # print(quant_cfg)
    model = reconstruct(model, quant_cfg.quant)
    #model.cuda()
    initialized = False

    quant_train(model)
    quant_epoch = quant_cfg.train.max_epoch
    tbar = tqdm(range(quant_epoch * len(trainloader)), desc="Compressing")
    for epoch in range(quant_epoch):
        for i, batch in enumerate(trainloader):
            #original = batch['data'].cuda(non_blocking=True)
            original = batch['data']
            data_classify = transform(original)
        
            if quant_cfg.name == 'ptq':
                model(im_cla=data_classify)

            elif quant_cfg.name == 'adaround':
                if not initialized:
                    quant_train(model, quantized=True)
                    model(im_cla=data_classify)
                    adaround_modules, param_groups = adaround_params(model)
                    optim = build_optimizer(param_groups, quant_cfg)
                    initialized = True

                quant_train(model, quantized=False)
                orig_output = []
                hooks = register_forward_hook(model, orig_output)
                model(im_cla=data_classify)
                remove_hooks(hooks)

                quant_eval(model, quantized=True)
                quant_output = []
                hooks = register_forward_hook(model, quant_output)
                model(im_cla=data_classify)
                remove_hooks(hooks)

                recon_loss = 0.
                for orig, quant in zip(orig_output, quant_output):
                    recon_loss += F.mse_loss(quant, orig.detach())
                
                beta = get_beta(epoch * len(trainloader) + i, quant_epoch * len(trainloader))
                reg_loss = 0.
                for module in adaround_modules:
                    reg_loss += module.regularization(beta)
                
                loss = recon_loss + reg_loss
                optim.zero_grad()
                loss.backward()
                optim.step()

            tbar.update()
    tbar.close()
    quant_eval(model, quantized=True)
    return model

    
def quantize_pack(model):
    for module in tqdm(model.modules(), desc='packing'):
        name = module.__class__.__name__
        if name in ['QuantLinear', 'QuantConv2d']:
            module.pack()
    # best_model_dict = deepcopy(model.state_dict())
    return model