import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .resnet import *
import math
from ncdia.augmentations import fantasy

class SAVCNET(nn.Module):

    def __init__(self, args, trans=1):
        super().__init__()
        if args.CIL.fantasy is not None:
            transform, trans = fantasy.__dict__[args.CIL.fantasy]()
        else:
            transform, trans = None, 0

        self.mode = args.network.net_savc.base_mode
        self.args = args

        self.encoder_q = resnet18(True, args, num_classes=self.args.network.net_savc.moco_dim)
        self.encoder_k = resnet18(True, args, num_classes=self.args.network.net_savc.moco_dim)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
        self.num_features = 512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.args.network.num_classes*trans, bias=False)
        self.att_fc = nn.Linear(self.num_features, self.args.network.att_classes, bias=False)
            
        self.K = self.args.network.net_savc.moco_k
        self.m = self.args.network.net_savc.moco_m
        self.T = self.args.network.net_savc.moco_t
        
        if self.args.network.net_savc.mlp:  # hack: brute-force replacement
            self.encoder_q.fc = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        # create the queue
        self.register_buffer("queue", torch.randn(self.args.network.net_savc.moco_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("label_queue", torch.zeros(self.K).long() - 1)

                    
    @torch.no_grad()
    def _momentum_update_key_encoder(self, base_sess):
        """
        Momentum update of the key encoder
        """
        if base_sess:
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        else:
            for k, v in self.encoder_q.named_parameters():
                if k.startswith('fc') or k.startswith('layer4') or k.startswith('layer3'):
                    self.encoder_k.state_dict()[k].data = self.encoder_k.state_dict()[k].data * self.m + v.data * (1. - self.m)

    #会把新的batchsize不断的放入到这个队列中，一开始的队列都是随机初始化的，应该是过了两个epoch之后可以是全部是图片样本了。
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
 
        # replace the keys and labels at ptr (dequeue and enqueue)
        if ptr + batch_size > self.K:
            remains = ptr + batch_size - self.K
            self.queue[:, ptr:] = keys.T[:, :batch_size - remains]
            self.queue[:, :remains] = keys.T[:, batch_size - remains:]
            self.label_queue[ptr:] = labels[ :batch_size - remains]
            self.label_queue[ :remains] = labels[batch_size - remains:]
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T # this queue is feature queue
            self.label_queue[ptr:ptr + batch_size] = labels        
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr
    
    def forward_metric(self, x):
        y, _ = self.encode_q(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(y, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.network.net_savc.temperature * x
            # x_att = F.linear(F.normalize(y, p=2, dim=-1), F.normalize(self.att_fc.weight, p=2, dim=-1))
            # x_att = self.args.network.net_savc.temperature * x_att
            x_att = self.att_fc(y)

        elif 'dot' in self.mode:
            x = self.fc(y)
            x_att = self.att_fc(y)

        return x, x_att # joint, contrastive

    def encode_q(self, x):
        x, y = self.encoder_q(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x, y
    
    def encode_k(self, x):
        x, y = self.encoder_k(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x, y

    def forward(self, im_cla, im_q=None, im_k=None, labels=None, im_q_small=None, base_sess=True, 
                last_epochs_new=False):
        if self.mode != 'encoder':
            if im_q == None:
                x, x_att = self.forward_metric(im_cla)
                return x, x_att
            else:  ##############  这里的没有加入属性的代码，小样本增量过程中需要小样本的再训练属性？？？ #########
                b = im_q.shape[0]
                logits_classify, logits_att = self.forward_metric(im_cla)
                #q,k都是过了一层fc层将特征降维
                _, q = self.encode_q(im_q)

                q = nn.functional.normalize(q, dim=1)
                feat_dim = q.shape[-1]
                q = q.unsqueeze(1) # bs x 1 x dim

                if im_q_small is not None:
                    _, q_small = self.encode_q(im_q_small)
                    q_small = q_small.view(b, -1, feat_dim)  # bs x 4 x dim
                    q_small = nn.functional.normalize(q_small, dim=-1)

                # compute key features
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder(base_sess)  # update the key encoder
                    _, k = self.encode_k(im_k)  # keys: bs x dim
                    k = nn.functional.normalize(k, dim=1)

                # compute logits
                # Einstein sum is more intuitive
                # positive logits: Nx1
                q_global = q
                l_pos_global = (q_global * k.unsqueeze(1)).sum(2).view(-1, 1)
                l_pos_small = (q_small * k.unsqueeze(1)).sum(2).view(-1, 1)

                # negative logits: NxK
                l_neg_global = torch.einsum('nc,ck->nk', [q_global.view(-1, feat_dim), self.queue.clone().detach()])
                l_neg_small = torch.einsum('nc,ck->nk', [q_small.view(-1, feat_dim), self.queue.clone().detach()])

                # logits: Nx(1+K)
                logits_global = torch.cat([l_pos_global, l_neg_global], dim=1)
                logits_small = torch.cat([l_pos_small, l_neg_small], dim=1)

                # apply temperature
                logits_global /= self.T
                logits_small /= self.T

                # one-hot target from augmented image
                positive_target = torch.ones((b, 1)).cuda()
                # find same label images from label queue
                # for the query with -1, all 
                targets = ((labels[:, None] == self.label_queue[None, :]) & (labels[:, None] != -1)).float().cuda()
                targets_global = torch.cat([positive_target, targets], dim=1)
                targets_small = targets_global.repeat_interleave(repeats=self.args.network.num_crops[1], dim=0)
                labels_small = labels.repeat_interleave(repeats=self.args.network.num_crops[1], dim=0)

                # dequeue and enqueue
                if base_sess or (not base_sess and last_epochs_new):
                    self._dequeue_and_enqueue(k, labels)
                
                return logits_classify, logits_global, logits_small, targets_global, targets_small          
        
        elif self.mode == 'encoder':  # 用来替换原型fc的
            x, _ = self.encode_q(im_cla)
            return x
        else:
            raise ValueError('Unknown mode')
        
    def inference(self, im_cla, **kwargs):
        # test_class, m 
        test_class = kwargs.get('test_class', None)
        m = kwargs.get('m', None)
        if test_class is None or m is None:
            raise ValueError("test_class and m parameters must be provided.")

        if self.mode != 'encoder':
            joint_preds, joint_preds_att = self.forward_metric(im_cla)
            joint_preds = joint_preds[:, :test_class*m]
            agg_preds = 0
            agg_preds_att = 0
            for j in range(m):
                agg_preds = agg_preds + joint_preds[j::m, j::m] / m
                agg_preds_att = agg_preds_att + joint_preds_att[j::m, :] / m

            return agg_preds, agg_preds_att
        elif self.mode == 'encoder':  
            x, _ = self.encode_q(im_cla)
            return x
        else:
            raise ValueError('Unknown mode')            
    
    def update_fc(self, dataloader, class_list, transform, session):
        for batch in dataloader:
            data, label, attribute = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b 
            labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
            data, _ =self.encode_q(data)
            data.detach()

        if self.args.network.net_savc.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list)*m, self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, labels, class_list, m)
            

    def update_fc_avg(self, data, labels, class_list, m):
        new_fc=[]
        for class_index in class_list:
            for i in range(m):
                index = class_index*m + i
                data_index = (labels==index).nonzero().squeeze(-1)
                embedding = data[data_index]
                proto = embedding.mean(0)
                new_fc.append(proto)
                self.fc.weight.data[index] = proto
        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    def get_logits(self, x, fc):
        if 'dot' in self.args.network.net_savc.new_mode:
            return F.linear(x, fc)
        elif 'cos' in self.args.network.net_savc.new_mode:
            return self.args.network.net_savc.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
    
    def get_features(self, x):
        x, y = self.encoder_q(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x
    