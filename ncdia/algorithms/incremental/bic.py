import logging
import numpy as np 
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from ncdia.utils import ALGORITHMS
from ncdia.algorithms.base import BaseAlg
from ncdia.utils.losses import AngularPenaltySMLoss
from ncdia.utils.metrics import accuracy, per_class_accuracy
from .hooks import BiCHook

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))
    def forward(self, x):
        return self.alpha * x + self.beta
    def printParam(self, i):
        print(i, self.alpha.item(), self.beta.item())

@ALGORITHMS.register
class BiC(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.args = trainer.cfg
        self.trainer = trainer

        self._network = None
        self._old_network = None
        self.transform = None
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        self.hook = BiCHook()
        self.bias_layers = nn.ModuleList([BiasLayer()] * (self.args.CIL.sessions + 1))
        self.T = 2
        trainer.register_hook(self.hook)

    def bias_forward(self, x):
        out = []
        b = self.args.CIL.base_classes
        w = self.args.CIL.way
        for s in range(self.args.CIL.sessions):
            if s > 0:
                out.append(self.bias_layers[s](x[:, b + (s - 1) * w:b + s * w]))
            else:
                out.append(self.bias_layers[s](x[:, :b]))
        return torch.cat(out, dim=1)

    def train_step(self, trainer, data, label, attribute, imgpath):
        """
        base train for bic method
        Args:
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """
        session = trainer.session
        self._network = trainer.model
        self._network.train()
        if session > 0:
            self._old_network = trainer.old_model
            self._old_network = self._old_network.cuda()
            self._old_network.eval()
            loss, acc, per_acc = self.loss_process(data, label, session, distill=True)
        else:
            loss, acc, per_acc = self.loss_process(data, label, session, distill=False)
        ret = {'loss': loss, 'acc': acc, 'per_class_acc': per_acc}
        return ret

    def val_step(self, trainer, data, label, *args, **kwargs):
        """Validation step for bic method.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Validation results. Contains the following:
                - "feature" (numpy.array): features in a batch
                - "logits" (numpy.array): logits in a batch
                - "predicts" (numpy.array): predicts in a batch
                - "confidence" (numpy.array): confidence in a batch
                - "label" (numpy.array): labels in a batch
                - "loss": Loss value.
                - "acc": Accuracy value.
        """
        session = trainer.session
        self._network = trainer.model
        self._network.eval()
        for _ in range(len(self.bias_layers)):
            self.bias_layers[_].train()
        loss, acc, per_acc = self.loss_process(data, label, session, distill=False)
        ret = {'loss': loss.item(), 'acc': acc.item(), 'per_class_acc': per_acc}
        return ret

    def test_step(self, trainer, data, label, *args, **kwargs):
        session = trainer.session
        self._network = trainer.model
        self._network.eval()
        loss, acc, per_acc = self.loss_process(data, label, session, distill=False)
        ret = {'loss': loss.item(), 'acc': acc.item(), 'per_class_acc': per_acc}
        return ret

    def get_net(self):
        return self._network

    def loss_process(self, data, label, session, distill=False):
        known_class = self.args.CIL.base_classes + session * self.args.CIL.way
        total_class = known_class + self.args.CIL.way
        data = data.cuda()
        labels = label.cuda()
        logits = self._network(data)
        logits = self.bias_forward(logits)
        logits_ = logits[:, :known_class]
        acc = accuracy(logits_, labels)[0]
        per_acc = str(per_class_accuracy(logits_, labels))
        clf_loss = self.loss(logits_, labels)
        if distill:
            lamda = known_class / total_class
            with torch.no_grad():
                old_logits = self._old_network(data)
                old_logits = self.bias_forward(old_logits)
                old_logits_ = old_logits[:, :known_class]
                old_logits = F.softmax(old_logits_ / self.T, dim=1)
            log_logits = F.log_softmax(old_logits_ / self.T, dim=1)
            distill_loss = -torch.mean(torch.sum(old_logits * log_logits, dim=1))
            loss = distill_loss * lamda + clf_loss * (1 - lamda)
        else:
            loss = clf_loss
        loss.backward()
        return loss, acc, per_acc
