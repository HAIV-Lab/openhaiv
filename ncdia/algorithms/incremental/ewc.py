import os
import torch
import numpy as np

from ncdia.utils import ALGORITHMS
from ncdia.algorithms.base import BaseAlg
from ncdia.utils.losses import AngularPenaltySMLoss
from ncdia.utils.metrics import accuracy, per_class_accuracy
from .hooks import EWCHook

@ALGORITHMS.register
class EWC(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.args = trainer.cfg
        self.trainer = trainer

        self._network = None
        self.transform = None
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        self.hook = EWCHook()
        trainer.register_hook(self.hook)
        session = trainer.session

        self.fisher = None
    
    def train_step(self, trainer, data, label, attribute, imgpath):
        """
        train for ewc method
        Args:
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """

        session = self.trainer.session
        known_class = self.args.CIL.base_classes + session * self.args.CIL.way
        self._network = trainer.model
        self._network.train()

        
        data = data.cuda()
        labels = label.cuda()
        logits = self._network(data)['logits']
        logits_ = logits[:, :known_class]
        if session >=1:
            self.mean = torch.load(os.path.join(trainer.work_dir, 'mean task_ ' + str(trainer.session - 1) + '.pth'))
            self.fisher = torch.load(os.path.join(trainer.work_dir, 'fisher task_ ' + str(trainer.session - 1) + '.pth'))
            loss_ewc = self.compute_ewc()
            
        acc = accuracy(logits_, labels)[0]
        per_acc = str(per_class_accuracy(logits_, labels))
        loss = self.loss(logits_, labels)

        if session>=1:
            loss = loss + 3.0 * loss_ewc
        loss.backward()

        ret = {}
        ret['loss'] = loss
        ret['acc'] =  acc
        ret['per_class_acc'] = per_acc

        return ret
        
    def val_step(self, trainer, data, label, *args, **kwargs):
        """
        Validation step for standard supervised learning.

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
        session = self.trainer.session
        test_class = self.args.CIL.base_classes + session * self.args.CIL.way
        self._network = trainer.model
        self._network.eval()
        data = data.cuda()
        labels = label.cuda()
        logits = self._network(data)['logits']
        logits_ = logits[:, :test_class]
        acc = accuracy(logits_, labels)[0]
        loss = self.loss(logits_, labels)
        per_acc = str(per_class_accuracy(logits_, labels))
        
        ret = {}
        ret['loss'] = loss.item()
        ret['acc'] = acc.item()
        ret['per_class_acc'] = per_acc
        
        return ret


    def test_step(self, trainer, data, label, *args, **kwargs):
        return self.val_step(trainer, data, label, *args, **kwargs)

    def compute_ewc(self):
        loss = 0 
        for n, p in self._network.named_parameters():
            if n in self.fisher.keys():
                loss += (
                        torch.sum(
                            (self.fisher[n])
                            * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                        )
                        / 2
                    )
        return loss

    