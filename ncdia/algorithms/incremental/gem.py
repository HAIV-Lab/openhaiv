import torch
import numpy as np
import quadprog

from ncdia.utils import ALGORITHMS
from ncdia.algorithms.base import BaseAlg
from ncdia.utils.losses import AngularPenaltySMLoss
from ncdia.utils.metrics import accuracy, per_class_accuracy
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
from ncdia.models.net.alice_net import AliceNET

import os
import logging
from tqdm import tqdm
import itertools
from torch import optim
import copy
from torch.utils.data import DataLoader
from ncdia.dataloader import MergedDataset
from ncdia.dataloader import BaseDataset


@HOOKS.register
class GEMHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()

    def after_train(self, trainer) -> None:
        filename = "task_" + str(trainer.session) + ".pth"
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))


@ALGORITHMS.register
class GEM(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.args = trainer.cfg
        self.trainer = trainer

        self._network = None
        self._old_network = None
        self.transform = None
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        # self.loss = AngularPenaltySMLoss(loss_type='cosface').cuda()
        self.hook = GEMHook()
        trainer.register_hook(self.hook)

        self.grad_dims = []
        self.margin = 0.5
        self.n_memories = 256
        for param in trainer.model.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), self.args.CIL.sessions).cuda()

        self.memory_data = torch.FloatTensor(
            self.args.CIL.sessions, self.n_memories, 3, 224, 224
        ).cuda()
        self.memory_label = torch.LongTensor(
            self.args.CIL.sessions, self.n_memories
        ).cuda()
        self.memory_pointer = 0

    def train_step(self, trainer, data, label, attribute, imgpath):
        """
        base train for Gradient Episodic Memory method
        Args:
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """
        session = self.trainer.session
        self._network = trainer.model
        if session > 0:
            self.update_memory(data, label, session)

        self._network.train()

        for past_session in range(session):
            self._network.zero_grad()
            inf, sup = self.calc_range(past_session)
            mask = torch.where(
                (self.memory_label[past_session] >= inf)
                & (self.memory_label[past_session] < sup)
            )[0].cuda()
            if len(mask) > 0:
                data_ = self.memory_data[past_session][mask]
                label_ = self.memory_label[past_session][mask]
                _ = self.forward(data_, label_, past_session, train=True, current=False)
                self.store_grad(past_session)

        self._network.zero_grad()
        inf, sup = self.calc_range(session)
        mask = torch.where((label >= inf) & (label < sup))[0].cuda()
        if len(mask) > 0:
            data_ = (data.cuda())[mask]
            label_ = (label.cuda())[mask]
            _ = self.forward(data_, label_, session, train=True, current=True)

        if session > 0:
            self.store_grad(session)
            dotp = torch.mm(
                self.grads[:, session].unsqueeze(0), self.grads[:, :session]
            )
            if (dotp < 0).sum() != 0:
                new_grad = self.project2cone2(session)
                self.overwrite_grad(new_grad)

        loss, acc, per_acc = self.forward(
            data, label, session, train=False, current=True
        )
        ret = {"loss": loss.item(), "acc": acc.item(), "per_class_acc": per_acc}

        return ret

    def val_step(self, trainer, data, label, *args, **kwargs):
        """Validation step for Gradient Episodic Memory.

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
        self._network = trainer.model
        self._network.eval()
        loss, acc, per_acc = self.forward(data, label, session, train=False)
        ret = {"loss": loss.item(), "acc": acc.item(), "per_class_acc": per_acc}
        return ret

    def test_step(self, trainer, data, label, *args, **kwargs):
        return self.val_step(trainer, data, label, *args, **kwargs)

    def get_net(self):
        return self._network

    def calc_range(self, session):
        sup = self.args.CIL.base_classes + session * self.args.CIL.way
        if session == 0:
            inf = 0
        else:
            inf = sup - self.args.CIL.way
        return inf, sup

    def forward(self, data, label, session, train=True, current=False):
        inf, sup = self.calc_range(session)
        data = data.cuda()
        labels = label.cuda()
        logits = self._network(data)
        if train:
            logits[:, :inf].data.fill_(-10e10)
            if not current:
                logits[:, sup:].data.fill_(-10e10)
        logits_ = logits[:, :sup]
        acc = accuracy(logits_, labels)[0]
        per_acc = str(per_class_accuracy(logits_, labels))
        loss = self.loss(logits_, labels)
        if train:
            loss.backward()
        return loss, acc, per_acc

    def update_memory(self, data, label, session):
        batchsize = label.data.size(0)
        start = self.memory_pointer
        end = min(start + batchsize, self.n_memories)
        inc = end - start
        self.memory_data[session, start:end].copy_(data.data[:inc])
        if batchsize == 1:
            self.memory_label[session, start] = label.data[0]
        else:
            self.memory_label[session, start:end].copy_(label.data[:inc])
        self.memory_pointer += batchsize
        if self.memory_pointer >= self.n_memories:
            self.memory_pointer = 0

    def store_grad(self, session):
        """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
        """
        # store the gradients
        self.grads[:, session].fill_(0.0)
        cnt = 0
        for param in self._network.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
                en = sum(self.grad_dims[: cnt + 1])
                self.grads[beg:en, session].copy_(param.grad.data.view(-1))
            cnt += 1

    def overwrite_grad(self, newgrad):
        """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
        """
        print("Overwriting Gradient......")
        cnt = 0
        for param in self._network.parameters():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
                en = sum(self.grad_dims[: cnt + 1])
                this_grad = newgrad[beg:en].contiguous().view(param.grad.data.size())
                param.grad.data.copy_(this_grad)
            cnt += 1

    def project2cone2(self, session):
        """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
        """
        old_grad = self.grads[:, :session].cpu().t().double().numpy()
        cur_grad = self.grads[:, session].cpu().contiguous().view(-1).double().numpy()
        C = old_grad @ old_grad.T
        p = old_grad @ cur_grad
        A = np.eye(old_grad.shape[0])
        b = np.zeros(old_grad.shape[0]) + self.margin
        v = quadprog.solve_qp(C, -p, A, b)[0]
        new_grad = old_grad.T @ v + cur_grad
        new_grad = torch.tensor(new_grad).float().cuda()
        return new_grad
