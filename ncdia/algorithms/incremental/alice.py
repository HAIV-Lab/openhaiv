import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ncdia.utils.cfg import Configs
from ncdia.utils.logger import Logger
from ncdia.utils import INMETHODS
from ncdia.utils import ALGORITHMS
from ncdia.models.resnet.resnet_models import *
from .base import BaseLearner
from .losses.angular_loss import AngularPenaltySMLoss
from .net.alice_net import AliceNET




    

@ALGORITHMS.register
class Alice(BaseLearner):
    def __init__(self, cfg: Configs) -> None:
        self.args = cfg.copy()
        super().__init__(self.args)

        self._network = AliceNET(self.args)
        self.loss = AngularPenaltySMLoss().cuda()


    
    def _init_train(self, data, label, attribute, imgpath):
        """
        base train for fact method
        Args:
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """
        self._network.train()
        
        masknum = 3
        mask=np.zeros((self.args.CIL.base_class,self.args.CIL.num_classes))
        for i in range(self.args.CIL.num_classes-self.args.CIL.base_class):
            picked_dummy=np.random.choice(self.args.CIL.base_class,masknum,replace=False)
            mask[:,i+self.args.CIL.base_class][picked_dummy]=1
        mask=torch.tensor(mask).cuda()


        data = data.cuda()
        labels = label.cuda()


        logits = self.net(data)
        logits_ = logits[:, :self.config.dataloader.base_class]
        pred = F.softmax(logits_, dim=1)
        acc = self._accuracy(labels, pred)
        loss = self.loss(logits_, labels)
        
        ret = {}
        ret['loss'] = loss
        ret['acc'] = acc

        return ret


    def _accuracy(self, labels, preds):
        """
        compute accuracy 
        Args:
            labels: true label
            preds: predict label
        """
        correct = (preds == labels).sum().item()  # 统计预测正确的数量
        total = labels.size(0)  # 总样本数量
        acc = correct / total  # 计算 accuracy
        return acc

    def _incremental_train(self):
        pass

    def get_net(self):
        return self._network