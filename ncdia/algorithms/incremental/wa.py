import torch
import numpy as np

from ncdia.utils import ALGORITHMS
from ncdia.algorithms.base import BaseAlg
from ncdia.utils.losses import AngularPenaltySMLoss
from ncdia.utils.metrics import accuracy, per_class_accuracy
from .hooks import WAHook


@ALGORITHMS.register
class WA(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.args = trainer.cfg
        self.trainer = trainer

        self._network = None
        self.transform = None
        self.loss = AngularPenaltySMLoss(loss_type='cosface').cuda()
        self.hook = WAHook()
        trainer.register_hook(self.hook)

        session = trainer.session

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
        known_class = self.args.CIL.base_classes + session * self.args.CIL.way
        self._network = trainer.model
        if session>=1:
            self._old_network = trainer.old_model
            self._old_network = self._old_network.cuda()
            self._old_network.eval()

        self._network.train()

        data = data.cuda()
        labels = label.cuda()
        logits = self._network(data)
        if session >=1:
            with torch.no_grad():
                old_logits = self._old_network(data)
        logits_ = logits[:, :known_class]
        acc = accuracy(logits_, labels)[0]
        per_acc = str(per_class_accuracy(logits_, labels))
        loss = self.loss(logits_, labels)
        if session >=1:
            kd_loss = self._KD_loss(logits_, old_logits[:, :known_class], 1.0)
            loss = loss + 3.0 * kd_loss
        loss.backward()

        ret = {}
        ret['loss'] = loss
        ret['acc'] = acc
        ret['per_class_acc'] = per_acc

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
        test_class = self.args.CIL.base_classes + session  * self.args.CIL.way
        self._network = trainer.model
        self._network.eval()
        data = data.cuda()
        labels = label.cuda()
        logits = self._network(data)
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
    
    def get_net(self):
        return self._network

    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]