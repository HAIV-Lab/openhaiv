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

class FACTTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:
        self.net = net
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
        
    def train_epoch_base(self, epoch_idx):
        self.net.train()
        loss_avg = 0.0

        train_dataiter = iter(self.train_loader)
        masknum=3
        mask=np.zeros((self.config.CIL.base_class,self.config.CIL.num_classes))
        for i in range(self.config.CIL.num_classes-self.config.CIL.base_class):
            picked_dummy=np.random.choice(self.config.CIL.base_class,masknum,replace=False)
            mask[:,i+self.config.CIL.base_class][picked_dummy]=1
        mask=torch.tensor(mask).cuda()

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

            if epoch_idx>=self.config.network.net_fact.loss_iter:
                logits_masked = logits.masked_fill(F.one_hot(train_label, num_classes=model.module.pre_allocate) == 1, -1e9)
                logits_masked_chosen= logits_masked * mask[train_label]
                pseudo_label = torch.argmax(logits_masked_chosen[:,self.config.dataloader.base_class:], dim=-1) + self.config.dataloader.base_class
                #pseudo_label = torch.argmax(logits_masked[:,config.base_class:], dim=-1) + config.base_class
                loss2 = F.cross_entropy(logits_masked, pseudo_label)

                index = torch.randperm(data.size(0)).cuda()
                pre_emb1=net.pre_encode(data)
                mixed_data=beta*pre_emb1+(1-beta)*pre_emb1[index]
                mixed_logits=net.post_encode(mixed_data)

                newys=train_label[index]
                idx_chosen=newys!=train_label
                mixed_logits=mixed_logits[idx_chosen]

                pseudo_label1 = torch.argmax(mixed_logits[:,self.config.dataloader.base_class:], dim=-1) + self.config.dataloader.base_class # new class label
                pseudo_label2 = torch.argmax(mixed_logits[:,:self.config.dataloader.base_class], dim=-1)  # old class label
                loss3 = F.cross_entropy(mixed_logits, pseudo_label1)
                novel_logits_masked = mixed_logits.masked_fill(F.one_hot(pseudo_label1, num_classes=model.module.pre_allocate) == 1, -1e9)
                loss4 = F.cross_entropy(novel_logits_masked, pseudo_label2)
                total_loss = loss+self.config.network.net_fact.balance*(loss2+loss3+loss4)
            else:
                total_loss = loss

            loss = total_loss
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
                # data = self.transform(data)
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

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced









