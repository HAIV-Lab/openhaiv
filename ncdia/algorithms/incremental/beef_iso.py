import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import itertools

from ncdia.utils import ALGORITHMS
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
from ncdia.algorithms.base import BaseAlg
from ncdia.dataloader import MergedDataset
from ncdia.dataloader import BaseDataset
from torch.utils.data import DataLoader
from ncdia.models.net.alice_net import AliceNET
from ncdia.utils.losses import AngularPenaltySMLoss
from ncdia.utils.metrics import accuracy, per_class_accuracy

EPSILON = 1e-8

@HOOKS.register
class BeefIsoHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
        self._fix_memory = True
    def before_train(self, trainer) -> None:
        trainer.train_loader
        _hist_trainset = trainer.hist_trainset
        now_dataset = trainer.train_loader.dataset
        _hist_trainset = MergedDataset([_hist_trainset], replace_transform=True)    

        if self._fix_memory and trainer.session >=1:
            _hist_trainset , new_dataset = self.construct_exemplar_unified(_hist_trainset, trainer.cfg.CIL.per_classes, trainer)
            _hist_trainset = MergedDataset([_hist_trainset], replace_transform=True)
            new_dataset = MergedDataset([new_dataset], replace_transform=True)

        if trainer.session >=1:
            _hist_trainset.merge([new_dataset], replace_transform=True)
        else:
            _hist_trainset.merge([now_dataset], replace_transform=True)
        # print(_hist_trainset.labels)
        trainer._train_loader = DataLoader(_hist_trainset, **trainer._train_loader_kwargs)

        # val_loader
        trainer.val_loader
        _hist_valset = MergedDataset([trainer.hist_valset], replace_transform=True)
        _hist_valset.merge([trainer.val_loader.dataset], replace_transform=True)
        trainer._val_loader = DataLoader(_hist_valset, **trainer._val_loader_kwargs)
        
        self._network.update_fc_before(self._total_classes)
        self._network_module_ptr = self._network
        
        if self.trainer.session > 0:
            for id in range(self.trainer.session):
                for p in self._network.convnets[id].parameters():
                    p.requires_grad = False
            for p in self._network.old_fc.parameters():
                p.requires_grad = False
    
    def after_train(self, trainer) -> None:
        # self._network.update_fc_before(self._total_classes)
        # self._network_module_ptr = self._network      
        self._network_module_ptr.update_fc_after()
        self._known_classes = self._total_classes
        if self.reduce_batch_size:
            if self._cur_task == 0:
                self.args["batch_size"] = self.args["batch_size"]
            else:
                self.args["batch_size"] = self.args["batch_size"] * (self._cur_task+1) // (self._cur_task+2) 



@ALGORITHMS.register
class BEEFISO(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.args = trainer.cfg
        self.trainer = trainer

        self._network = None
        self.transform = None
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        self.hook = BeefIsoHook()
        trainer.register_hook(self.hook)
        session = trainer.session
        self.sinkhorn_reg = self.args.sinkhorn
        self.calibration_term = self.args.calibration_term
    
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
        self._network = trainer.model
        self._network.train()

        data = data.cuda()
        labels = label.cuda()
        loss, acc, per_acc = self._compute_loss(self, data, labels)
        loss.backward()

        ret = {'loss': loss, 'acc': acc, 'per_class_acc': per_acc}
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

        ret = {'loss': loss.item(), 'acc': acc.item(), 'per_class_acc': per_acc}
        
        return ret
    
    def test_step(self, trainer, data, label, *args, **kwargs):
        return self.val_step(trainer, data, label, *args, **kwargs)

    
    def target2onehot(self, targets, n_classes):
        onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
        onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
        return onehot
    
    def _compute_loss(self, data, labels):
        session = self.trainer.session
        if session == 0:
            known_class = self.args.CIL.base_classe
        else:
            known_class = self.args.CIL.base_classes + (session - 1) * self.args.CIL.way
        total_class = self.args.CIL.base_classes + session * self.args.CIL.way
        
        logits = self._network(data)["logits"]
        logits_ = logits[:, :total_class]
        loss_en = self.args["energy_weight"] * self.get_energy_loss(data,labels,labels)
        loss = F.cross_entropy(data, labels)
        loss = loss + loss_en
        
        acc = accuracy(logits_, labels)[0]
        per_acc = str(per_class_accuracy(logits_, labels))

        return loss, acc, per_acc
    
    def get_energy_loss(self,inputs,targets,pseudo_targets):
        inputs = self.sample_q(inputs)
        
        out = self._network(inputs)
        if self._cur_task == 0:
            targets = targets + self._total_classes
            train_logits, energy_logits = out["logits"], out["energy_logits"]
        else:
            targets = targets + (self._total_classes - self._known_classes) + self._cur_task
            train_logits, energy_logits = out["train_logits"], out["energy_logits"]
        
        logits = torch.cat([train_logits,energy_logits],dim=1)
        
        logits[:,pseudo_targets] = 1e-9        
        energy_loss = F.cross_entropy(logits,targets)
        return energy_loss
    
    def sample_q(self, replay_buffer, n_steps=3):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        self._network_copy = self._network_module_ptr.copy().freeze()
        init_sample = replay_buffer
        init_sample = torch.rot90(init_sample, 2, (2, 3))
        embedding_k = init_sample.clone().detach().requires_grad_(True)
        optimizer_gen = torch.optim.SGD(
            [embedding_k], lr=1e-2)
        for k in range(1, n_steps + 1):
            out = self._network_copy(embedding_k)
            if self._cur_task == 0:
                energy_logits, train_logits = out["energy_logits"], out["logits"]
            else:
                energy_logits, train_logits = out["energy_logits"], out["train_logits"]
            num_forwards = energy_logits.shape[1]
            logits = torch.cat([train_logits,energy_logits],dim=1)
            negative_energy = torch.log(torch.sum(torch.softmax(logits,dim=1)[:,-num_forwards:]))
            optimizer_gen.zero_grad()
            negative_energy.sum().backward()
            optimizer_gen.step()
            embedding_k.data += 1e-3 * \
                torch.randn_like(embedding_k)
        final_samples = embedding_k.detach()
        return final_samples