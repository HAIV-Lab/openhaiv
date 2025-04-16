import os
import torch
import numpy as np
from torch import optim

from ncdia.utils import ALGORITHMS
from ncdia.algorithms.base import BaseAlg
from ncdia.utils.losses import AngularPenaltySMLoss
from ncdia.utils.metrics import accuracy, per_class_accuracy
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import QuantifyHook
from ncdia.models.net.inc_net import IncrementalNet

@HOOKS.register
class EWCHook(QuantifyHook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        filename = 'task_' + str(trainer.session) + '.pth'
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))
        self.update_fisher(trainer)
    
    def update_fisher(self, trainer):
        session = trainer.session
        if session == 0:
            fisher = self.getFisherDiagonal(trainer.train_loader, trainer)
            filename = 'fisher task_ ' + str(trainer.session) + '.pth'
            torch.save(fisher, os.path.join(trainer.work_dir, filename))
        else:
            known_classes = trainer.cfg.CIL.base_classes + session * trainer.cfg.CIL.way
            alpha = known_classes / trainer.cfg.CIL.num_classes
            fisher = self.fisher = torch.load(os.path.join(trainer.work_dir, 'fisher task_ ' + str(trainer.session - 1) + '.pth'))
            new_fisher = self.getFisherDiagonal(trainer.train_loader, trainer)
            for n, p in new_fisher.items():
                new_fisher[n][: len(fisher[n])] = (
                    alpha * fisher[n]
                    + (1 - alpha) * new_fisher[n][: len(fisher[n])]
                )
            fisher = new_fisher
            filename = 'fisher task_ ' + str(trainer.session) + '.pth'
            torch.save(fisher, os.path.join(trainer.work_dir, filename))
        mean = {
            n: p.clone().detach()
            for n, p in trainer.model.named_parameters()
            if p.requires_grad
        }
        mean_name = 'mean task_ ' + str(trainer.session) + '.pth'
        torch.save(mean, os.path.join(trainer.work_dir, mean_name))
 

        
    def getFisherDiagonal(self, train_loader, trainer):
        fisher = {
            n: torch.zeros(p.shape).cuda()
            for n, p in  trainer.model.named_parameters()
            if p.requires_grad
        }

        trainer.model.train()
        optimizer = optim.SGD(trainer.model.parameters(), lr=0.1)
        for i, batch in enumerate(train_loader):
            inputs = batch['data'].cuda()
            targets = batch['label'].cuda()
            logits = trainer.model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            for n, p in trainer.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(0.0001))

        return fisher

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
        logits = self._network(data)
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

    