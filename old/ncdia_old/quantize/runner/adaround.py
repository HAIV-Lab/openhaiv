"""
AdaRound runner
version: 0.0.3
update: 2023-12-22
"""
from torch.nn import functional as F

from utils import accuracy
from modelzoo import RANGES
from optim import build_optimizer, build_lr_scheduler
from .ptq import PTQ


class AdaRound(PTQ):
    """ AdaRound: Adaptive Rounding for PTQ

    Up or Down? Adaptive Rounding for Post-Training Quantization
    https://arxiv.org/abs/2004.10568

    Args:
        cfg (Configs): configuration
        cfg.model (Configs): model configs
        cfg.model.name (str): name of model, registed in modelzoo
        cfg.runner (Configs): runner configs
        cfg.runner.name (str): name of runner, registed in runnerzoo, should be 'adaround'
        cfg.runner.beta (float): beta for regularization loss, default 10.
            'dynamic' uses linearly decreasing beta from 20 to 2
        cfg.runner.verbose (bool): whether to print verbose info, default False
        cfg.quant (Configs): quant configs
        cfg.optimizer.name (str): name of optimizer, registed in torch.optim
        cfg.optimizer.lr (float): learning rate
        cfg.lr_scheduler.name (str): name of lr_scheduler, registed in torch.optim.lr_scheduler
        cfg.train (Configs): training configs
        cfg.train.max_epoch (int): max number of epochs to train, default 1
        cfg.train.print_freq (int): frequency of printing training status, default 10

    """
    def __init__(self, cfg):
        self.check_cfg(cfg, 'adaround')
        super().__init__(cfg)
        self.initialized = False
        self.hooks = []

    def param_groups(self, model=None):
        """ Gather parameters of AdaRound modules.
        Args:
            model (nn.Module): model, default None (self.model)
        Returns:
            list: list of parameters
        """
        if model is None:
            model = self.model

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

    def build_optim(self):
        """ Build optimizer and lr_scheduler.
        """
        self.modules, param_groups = self.param_groups()
        self.optim = build_optimizer(param_groups, self.cfg)
        self.sched = build_lr_scheduler(self.optim, self.cfg)
    
    def register_forward_hook(self, quantized=False):
        """ Register forward hook to get output of each layer.
        Args:
            quantized (bool): if True, gather quantized output, 
                                else gather original output
        """
        def hook(module, _, output):
            name: str = module.__class__.__name__
            if not name.startswith('Quant') or name == 'Quantizer':
                return

            if not quantized:
                self.orig_output.append(output.detach().clone())
            else:
                self.quant_output.append(output)
        
        for module in self.model.modules():
            self.hooks.append(module.register_forward_hook(hook))

    def remove_hooks(self):
        """ Remove registered hooks.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_beta(self, current, total, start=20, end=2, warmup=0.2):
        """ Get beta for regularization loss.
            Higher beta allows h(V) to adapt freely in the initial phase,
            and lower beta encourages it to converge to 0 or 1 in the later phase.
        Args:
            current (int): current iteration
            total (int): total number of iterations
            start (int): start value of beta, default 20
            end (int): end value of beta, default 2
            warmup (float): ratio of warmup iterations, default 0.2
        Returns:
            float: beta
        """
        if current / total < warmup:
            return start
        else:
            return start + (end - start) * (current / total - warmup) / (1 - warmup)

    def train_step(self, batch, iters, **kwargs):
        images = batch["img"].to(self.device)
        labels = batch["label"].to(self.device)

        if not self.initialized:
            self.train(quantized=True)
            self.model(images)
            self.build_optim()
            self.initialized = True

        # forward pass with original model
        self.train(quantized=False)
        self.orig_output = []
        self.register_forward_hook(quantized=False)
        self.model(images)
        self.remove_hooks()

        # forward pass with quantized model
        self.eval(quantized=True)
        self.quant_output = []
        self.register_forward_hook(quantized=True)
        output = self.model(images)
        self.remove_hooks()

        # compute reconstruction and regularization loss
        recon_loss = 0.
        for orig, quant in zip(self.orig_output, self.quant_output):
            recon_loss += F.mse_loss(quant, orig)

        if self.cfg.runner.beta == 'dynamic':
            beta = self.get_beta(iters, self.total_iters)
        else:
            beta = self.cfg.runner.beta

        reg_loss = 0.
        for module in self.modules:
            reg_loss += module.regularization(beta)

        loss = recon_loss + reg_loss

        # update parameters
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item(), accuracy(output, labels)[0].item(), images.size(0)
