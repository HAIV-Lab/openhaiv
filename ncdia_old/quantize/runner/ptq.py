"""
PTQ
version: 0.1.0
update: 2023-12-22
"""
import torch
from torch.nn import functional as F

from utils import accuracy
from modelzoo import build_model, reconstruct
from modelzoo import Quantizer
from .base import BasicRunner


class PTQ(BasicRunner):
    """ Post Training Quantization (PTQ).

    A White Paper on Neural Network Quantization
    https://arxiv.org/abs/2106.08295

    Args:
        cfg (Configs): configuration
        cfg.model (Configs): model configs
        cfg.model.name (str): name of model, registed in modelzoo
        cfg.runner (Configs): runner configs
        cfg.runner.name (str): name of runner, registed in runnerzoo
        cfg.runner.verbose (bool): whether to print verbose info, default False
        cfg.quant (Configs): quant configs
        cfg.train (Configs): training configs
        cfg.train.max_epoch (int): max number of epochs to train, default 1
        cfg.train.print_freq (int): frequency of printing training status, default 10

    """
    def __init__(self, cfg):
        self.check_cfg(cfg, 'ptq')
        super().__init__(cfg)

        self.max_epoch = cfg.train.max_epoch or 1
        self.print_freq = cfg.train.print_freq or 10
        self.device = cfg.device or "cuda"
        self.verbose = cfg.runner.verbose

        self.model = build_model(cfg.model)
        self.model = reconstruct(self.model, cfg.quant)
        if self.verbose:
            cfg.logger.info(self.model)

        self.model.to(self.device)
        self.register_model('model', self.model)

    def train(self, train=True, quantized=False):
        """ Set model to calibration mode.
        Args:
            train (bool): whether to set model to training (calibrating) mode
            quantized (bool): whether to set model to quantized mode for inference
        """
        super().train(train)
        for model in self._models.values():
            for module in model.modules():
                if hasattr(module, 'calibrating'):
                    module.calibrating = train
                if isinstance(module, Quantizer):
                    module.quant(quantized)
    
    def eval(self, quantized=False):
        """ Set model to evaluation mode.
        """
        self.train(False, quantized)

    def train_step(self, batch, **kwargs):
        images = batch["img"].to(self.device)
        labels = batch["label"].to(self.device)

        output = self.model(images)
        loss = F.cross_entropy(output, labels)

        return loss.item(), accuracy(output, labels)[0].item(), images.size(0)

    def eval_step(self, batch, quantized=False, **kwargs):
        self.eval(quantized)

        with torch.no_grad():
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)

            output = self.model(images)
            loss = F.cross_entropy(output, labels)
        
        return loss.item(), accuracy(output, labels)[0].item(), images.size(0)

    def update(self, epoch):
        super().update(epoch)

        cfg = self.cfg
        eval_result = None

        if cfg.train.eval_freq and (epoch + 1) % cfg.train.eval_freq == 0:
            eval_result = self.evaluate(self.val_loader, quantized=True)
        
        if cfg.train.save_freq and (epoch + 1) % cfg.train.save_freq == 0:
            self.save_model(eval_result)

        if (epoch + 1) == self.max_epoch:
            eval_result = self.evaluate(self.val_loader, quantized=True)
            ### debug
            from tqdm import tqdm
            for model in self._models.values():
                for module in tqdm(model.modules(), desc='packing'):
                    name = module.__class__.__name__
                    if name in ['QuantLinear', 'QuantConv2d', 'QuantMultiheadAttention']:
                        module.pack()
            self.optim, self.sched = None, None
            ### debug end
            self.save_model(eval_result)
