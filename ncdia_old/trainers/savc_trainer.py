import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import ncdia_old.utils.comm as comm
from ncdia_old.utils import Config

from .lr_scheduler import cosine_annealing
from ncdia_old.augmentations import fantasy
from ncdia_old.losses.SupConv1 import SupContrastive

class SAVCTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config

        self.supcon_loss = SupContrastive().cuda() 

        self.new_fc = None
        self.old_class = -1
        self.new_class = -1

        if config.CIL.fantasy is not None:
            self.transform, self.num_trans = fantasy.__dict__[config.CIL.fantasy]()
        else:
            self.transform = None
            self.num_trans = 0

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(  # 适合在step中更新的
        #     self.optimizer,
        #     lr_lambda=lambda step: cosine_annealing(
        #         step,
        #         config.optimizer.num_epochs * len(train_loader),
        #         1,
        #         1e-6 / config.optimizer.lr,
        #     ),
        # )
        if self.config.optimizer.schedule == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.optimizer.step, \
                                                             gamma=self.config.optimizer.gamma)
        elif self.config.optimizer.schedule == 'Milestone':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.optimizer.milestones, \
                                                             gamma=self.config.optimizer.gamma)
        elif self.config.optimizer.schedule == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.optimizer.num_epochs)


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
            single_labels = batch['label']

            # ------- data preparation  --------- #
            b, c, h, w = data[1].shape
            original = data[0].cuda(non_blocking=True)
            data[1] = data[1].cuda(non_blocking=True)
            data[2] = data[2].cuda(non_blocking=True)
            single_labels = single_labels.cuda(non_blocking=True)

            if len(self.config.CIL.num_crops) > 1:
                data_small = data[self.config.CIL.num_crops[0]+1].unsqueeze(1)
                for j in range(1, self.config.CIL.num_crops[1]):
                    data_small = torch.cat((data_small, data[j+self.config.CIL.num_crops[0]+1].unsqueeze(1)), dim=1)
                data_small = data_small.view(-1, c, self.config.CIL.size_crops[1], \
                                             self.config.CIL.size_crops[1]).cuda(non_blocking=True)
            else:
                data_small = None

            data_classify = self.transform(original)    
            data_query = self.transform(data[1])
            data_key = self.transform(data[2])
            data_small = self.transform(data_small)

            m = data_query.size()[0] // b
            joint_labels = torch.stack([single_labels*m+ii for ii in range(m)], 1).view(-1)

            # ------  forward  ------- #
            joint_preds = self.net(im_cla=data_classify)  
            joint_preds = joint_preds[:, :self.config.CIL.base_class*m]
            joint_loss = F.cross_entropy(joint_preds, joint_labels)

            agg_preds = 0
            for i in range(m):
                agg_preds = agg_preds + joint_preds[i::m, i::m] / m

            loss = joint_loss

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        self.scheduler.step()
        lrc = self.scheduler.get_last_lr()[0]
        print('NOW Learning rate: ', lrc)
        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics
    
    def replace_base_fc(self, train_loader):
        """update the model's fc with the new trainloader
        """
        self.net.eval()
        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                # print(batch)
                data = batch['data'].cuda()
                label = batch['label'].cuda()
  
                b = data.size()[0]
                data = self.transform(data)
                m = data.size()[0] // b
                labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
                embedding = self.net.get_features(data)

                embedding_list.append(embedding.cpu())
                label_list.append(labels.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        proto_list = []

        for class_index in range(self.config.CIL.base_class*m):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)

        proto_list = torch.stack(proto_list, dim=0)

        self.net.fc.weight.data[:self.config.CIL.base_class*m] = proto_list

        return self.net

    def update_session_info(self, session):
        """new session training preparation.
           update the optimizer and schedule, only for new_fc
        """
        self.old_class = self.config.CIL.base_class + self.config.CIL.way * (session - 1)
        self.new_class = self.config.CIL.base_class + self.config.CIL.way * session 
        self.new_fc = nn.Parameter(
            torch.rand(self.config.CIL.way*self.num_trans, self.net.num_features, device="cuda"),
            requires_grad=True)
        self.new_fc.data.copy_(self.net.fc.weight[self.old_class*self.num_trans : self.new_class*self.num_trans, :].data)
        
        # ------ update the optimizer and schedule, only for new_fc ------- #
        self.optimizer = torch.optim.SGD([
            {'params': self.new_fc, 
             'lr': self.config.optimizer.lr_new},
            {'params': self.net.encoder_q.fc.parameters(), 
             'lr': 0.05 * self.config.optimizer.lr_new},
            {'params': self.net.encoder_q.layer4.parameters(), 
             'lr': 0.002 * self.config.optimizer.lr_new}, 
            ],
            momentum=0.9, dampening=0.9, weight_decay=0)
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                self.config.optimizer.new_epochs * len(self.train_loader),
                1,
                1e-6 / self.config.optimizer.lr,
            ),
        )

    def train_epoch_new(self, epoch_idx):
        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        assert self.new_fc is not None and self.old_class != -1 and self.new_class != -1, \
            "Need func(update_session_info) first!"

        with torch.enable_grad():
            for train_step in tqdm(range(1, len(train_dataiter) + 1),
                                desc='Epoch {:03d}: '.format(epoch_idx),
                                position=0,
                                leave=True,
                                disable=not comm.is_main_process()):
                batch = next(train_dataiter)
                data = batch['data']
                single_labels = batch['label']

                # ------- data preparation  --------- #
                b, c, h, w = data[1].shape
                original = data[0].cuda(non_blocking=True)
                data[1] = data[1].cuda(non_blocking=True)
                data[2] = data[2].cuda(non_blocking=True)
                single_labels = single_labels.cuda(non_blocking=True)
                if len(self.config.CIL.num_crops) > 1:
                    data_small = data[self.config.CIL.num_crops[0]+1].unsqueeze(1)
                    for j in range(1, self.config.CIL.num_crops[1]):
                        data_small = torch.cat((data_small, data[j+self.config.CIL.num_crops[0]+1].unsqueeze(1)), dim=1)
                    data_small = data_small.view(-1, c, self.config.CIL.size_crops[1], \
                                                 self.config.CIL.size_crops[1]).cuda(non_blocking=True)
                else:
                    data_small = None

                data_classify = self.transform(original)    
                data_query = self.transform(data[1])
                data_key = self.transform(data[2])
                data_small = self.transform(data_small)

                m = self.num_trans  # data_query.size()[0] // b  # self.num_trans
                joint_labels = torch.stack([single_labels*m+ii for ii in range(m)], 1).view(-1)


                # ------  forward without atttribute training  ------- #
                old_fc = self.net.fc.weight[:self.old_class*m, :].clone().detach()    
                fc = torch.cat([old_fc, self.new_fc], dim=0)  # 16 + 6

                features, _ = self.net.encode_q(data_classify)
                features.detach()
                logits = self.net.get_logits(features, fc)
                joint_labels = joint_labels.long()
                joint_loss = F.cross_entropy(logits, joint_labels)
 
                loss = joint_loss #+ loss_moco         

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # exponential moving average, show smooth values
                with torch.no_grad():
                    loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
