import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import ncdia.utils.comm as comm
from ncdia.utils import Config

from .lr_scheduler import cosine_annealing
from ncdia.augmentations import fantasy
from ncdia.losses.SupConv1 import SupContrastive


class LwFTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:
        self.net = net
        self._old_net = copy.deepcopy(self.net).freeze()
        self.train_loader = train_loader
        self.config = config

        self.loss = nn.CrossEntropyLoss().cuda()

        self.new_fc = None
        self.old_class = -1
        self.new_class = -1

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        if self.config.optimizer.schedule == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.optimizer.step, \
                                                             gamma=self.config.optimizer.gamma)
        elif self.config.optimizer.schedule == 'Milestone':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.optimizer.milestones, \
                                                             gamma=self.config.optimizer.gamma)
        elif self.config.optimizer.schedule == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.optimizer.num_epochs)
    
    def after_task(self):
        self._old_net = copy.deepcopy(self.net).freeze()

    def train_epoch_base(self, epoch_idx):
        self.net.train()
        loss_avg = 0.0

        train_dataiter = iter(self.train_loader)
        for train_step in tqdm(range(1,
                                len(train_dataiter) + 1),
                        desc='Epoch {:03d}: '.format(epoch_idx),
                        position=0,
                        leave=True,
                        disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data']
            labels = batch['label']

            data = data.cuda()
            labels = labels.cuda()


            embeddings = self.net.encode(data)
            logits = self.net(data)
            logits_ = logits[:, :self.config.dataloader.base_class]
            loss = self.loss(logits_, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        self.scheduler.step()
        lrc = self.scheduler.get_last_lr()[0]
        print('NOW Learning rate: ', lrc)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def inc_train(self, epoch_idx, session):
        self.net.train()
        loss_avg = 0.0

        train_dataiter = iter(self.train_loader)
        for train_step in tqdm(range(1,
                                len(train_dataiter) + 1),
                        desc='Epoch {:03d}: '.format(epoch_idx),
                        position=0,
                        leave=True,
                        disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data']
            labels = batch['label']

            data = data.cuda()
            labels = labels.cuda()


            embeddings = self.net.encode(data)
            logits = self.net(data)
            logits_ = logits[:, :self.config.dataloader.base_class + self.config.dataloader.way*session]
            loss = self.loss(logits_, labels)
            loss_kd = self._KD_loss(
                    logits[:, : self.config.dataloader.base_class],
                    self._old_net(data)[:, : self.config.dataloader.base_class],
                    2.0,
                )
            loss = loss + 3.0 * loss_kd

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        self.scheduler.step()
        lrc = self.scheduler.get_last_lr()[0]
        print('NOW Learning rate: ', lrc)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics
    
    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced

    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

