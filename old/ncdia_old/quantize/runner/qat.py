"""
QAT
version: 0.0.1
update: 2024-01-15
"""
from torch.nn import functional as F

from utils import accuracy
from optim import build_optimizer, build_lr_scheduler
from .base import BasicRunner
from .ptq import PTQ


class QAT(PTQ):
    """ Quantization Aware Training (QAT).

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
        cfg.train.calibrated_epoch (int): number of epochs to calibrate, default 1
        cfg.train.max_epoch (int): max number of epochs to train, default 1
        cfg.train.print_freq (int): frequency of printing training status, default 10

    """
    def __init__(self, cfg):
        self.check_cfg(cfg, 'qat')
        super().__init__(cfg)

        self.calibrated_epoch = cfg.train.calibrated_epoch or 1
        self.max_epoch += self.calibrated_epoch

        self.initialized = False

    def build_optim(self):
        """ Build optimizer and lr_scheduler.
        """
        for param in self.model.parameters():
            param.requires_grad = True
        self.optim = build_optimizer(self.model.parameters(), self.cfg)
        self.sched = build_lr_scheduler(self.optim, self.cfg)
    
    def train_step(self, batch, **kwargs):
        if not self.initialized:
            return super().train_step(batch, **kwargs)
        
        self.eval(quantized=True)
        images = batch["img"].to(self.device)
        labels = batch["label"].to(self.device)

        output = self.model(images)
        loss = F.cross_entropy(output, labels)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item(), accuracy(output, labels)[0].item(), images.size(0)
        
    def update(self, epoch):
        BasicRunner.update(self, epoch)

        if (epoch + 1) == self.calibrated_epoch:
            eval_result = self.evaluate(self.val_loader, quantized=True)
            # self.save_model(eval_result)

            self.build_optim()
            self.initialized = True

        else:
            cfg = self.cfg
            eval_result = None

            if (epoch + 1) == self.max_epoch:
                eval_result = self.evaluate(self.val_loader, quantized=True)
                ### debug
                # from tqdm import tqdm
                # for model in self._models.values():
                #     for module in tqdm(model.modules(), desc='packing'):
                #         name = module.__class__.__name__
                #         if name in ['QuantLinear', 'QuantConv2d', 'QuantMultiheadAttention']:
                #             module.pack()
                self.optim, self.sched = None, None
                ### debug end
                self.save_model(eval_result)
                return

            if cfg.train.eval_freq and (epoch + 1) % cfg.train.eval_freq == 0:
                eval_result = self.evaluate(self.val_loader, quantized=True)

            if cfg.train.save_freq and (epoch + 1) % cfg.train.save_freq == 0:
                self.save_model(eval_result)
