import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from ncdia.dataloader.augmentations import fantasy
from ncdia.utils import MODELS


@MODELS.register
class SAVCNET(nn.Module):
    def __init__(self, network, base_classes, num_classes, att_classes, net_savc, mode=None):
        super().__init__()
        if net_savc.fantasy is not None:
            self.transform, trans = fantasy.__dict__[net_savc.fantasy]()
        else:
            self.transform, trans = None, 0
        
        self.mode = net_savc.base_mode
        self.network = network
        

        network = network.cfg
        network['pretrained'] = True
        network['num_classes'] = 1000
        self.encoder_q = MODELS.build(network)
        self.num_features = 512
        self.base_classes = base_classes
        self.num_classes = num_classes
        self.att_classes = att_classes
        self.net_savc = net_savc

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.num_classes*trans, bias=False)
            
        if self.net_savc.mlp:  # hack: brute-force replacement
            self.encoder_q.fc = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU(), self.encoder_q.fc)


    def forward_metric(self, x):
        y, _ = self.encode_q(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(y, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.net_savc.temperature * x
        elif 'dot' in self.mode:
            x = self.fc(y)
        return x

    def encode_q(self, x):
        x, y = self.encoder_q(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x, y

    def forward(self, im_cla):
        if self.mode != 'encoder':
            x = self.forward_metric(im_cla)
            return x          
        
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
            joint_preds = self.forward_metric(im_cla)
            joint_preds = joint_preds[:, :test_class*m]
            agg_preds = 0

            for j in range(m):
                agg_preds = agg_preds + joint_preds[j::m, j::m] / m

            return agg_preds
        elif self.mode == 'encoder':  
            x, _ = self.encode_q(im_cla)
            return x
        else:
            raise ValueError('Unknown mode')

    def update_fc_savc(self, dataloader, class_list, session):
        embedding_list = []
        label_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                data = batch['data'].cuda()
                label = batch['label'].cuda()
                b = data.size()[0]
                data = self.transform(data)
                m = data.size()[0] // b
                labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
                embedding = self.get_features(data)

                embedding_list.append(embedding.cpu())
                label_list.append(labels.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        proto_list = []

        print(class_list)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)

        proto_list = torch.stack(proto_list, dim=0)
        print(proto_list.shape)
        print(proto_list)
        print(self.fc.weight.data.shape)
        input()
        self.fc.weight.data[11:] = proto_list

    def update_fc(self, dataloader, class_list, session):

        for batch in dataloader:
            data = batch['data'].cuda()
            label = batch['label'].cuda()
 
            b = data.size()[0]
            data = self.transform(data)
            m = data.size()[0] // b 
            labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
            # data, _ = self.encode_q(data)
            # data.detach()
            data = self.get_features(data)
            data.detach()


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
        if 'dot' in self.net_savc.new_mode:
            return F.linear(x, fc)
        elif 'cos' in self.model.net_savc.new_mode:
            return self.model.net_savc.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
    
    def get_features(self, x):
        x, y = self.encoder_q(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x
    