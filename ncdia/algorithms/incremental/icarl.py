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
from .hooks import iCaRLHook

@ALGORITHMS.register
class iCaRL(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.args = trainer.cfg
        self.trainer = trainer

        self._network = None
    
    def train_step(self, trainer, data, label, attribute, imgpath):
        """
        base train for fact method
        Args:
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """
        session = self.trainer.session

        if session == 0:
            self._network = trainer.model
            self._network.train()

            data = data.cuda()
            labels = label.cuda()

            logits = self._network(data)
            logits_ = logits[:, :self.args.CIL.base_classes]

            acc = accuracy(logits_, labels)[0]
            per_acc = str(per_class_accuracy(logits_, labels))
            loss = self.loss(logits_, labels)
            loss.backward()

            ret = {}
            ret['loss'] = loss
            ret['acc'] = acc
            ret['per_class_acc'] = per_acc
        else:
            pass
        
        return ret

    def val_step(self, trainer, data, label, *args, **kwargs):
        """Validation step for standard supervised learning.

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

        if session == 0:
            self._network = trainer.model
            self._network.eval()

            with torch.no_grad():
                data = data.cuda()
                labels = label.cuda()

                logits = self._network(data)
                logits_ = logits[:, :self.args.CIL.base_classes]
                acc = accuracy(logits_, labels)[0]
                loss = self.loss(logits_, labels)
                per_acc = str(per_class_accuracy(logits_, labels))
                
                ret = {}
                ret['loss'] = loss.item()
                ret['acc'] = acc.item()
                ret['per_class_acc'] = per_acc
            else:
                test_class = self.args.CIL.base_classes + session * self.args.CIL.way

                with torch.no_grad():
                    data = data.cuda()
                    labels = label.cuda()

                    b = data.size()[0]
                    joint_preds = self._network(data)
                    acc = accuracy(joint_preds, labels)[0]
                    loss = self.loss(agg_preds, labels)
                    per_acc = str(per_class_accuracy(agg_preds, labels))
                    
                    ret = {}
                    ret['loss'] = loss.item()
                    ret['acc'] = acc.item()
                    ret['per_class_acc'] = per_acc
                    
            return ret


    def test_step(self, trainer, data, label, *args, **kwargs):
        return self.val_step(trainer, data, label, *args, **kwargs)

    
    def get_net(self):
        return self._network