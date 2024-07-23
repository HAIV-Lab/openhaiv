import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ncdia.utils.cfg import Configs
from ncdia.utils.logger import Logger
from ncdia.utils import INMETHODS
from ncdia.algorithms.base import BaseAlg
# from ncdia.models.resnet.resnet_models import *
from .base import BaseLearner
from .net.fact_net import FACTNET
from ncdia.utils import ALGORITHMS
from ncdia.utils.metrics.accuracy import accuracy


    

@ALGORITHMS.register
class FACT(BaseAlg):
    def __init__(self, cfg: Configs) -> None:
        print("++++++++++cfg:", cfg)
        self.args = cfg.copy()
        # self.config = cfg
        super().__init__()

        self.base_class = 11
        # self._network = FACTNET(self.args)
        self._network = None
        self.loss = nn.CrossEntropyLoss().cuda()
        self.beta = 0.5
        


    
    def train_step(self, trainer, data, label, *args, **kwargs):
        """
        base train for fact method
        Args:
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """
        self._network = trainer.model
        self._network.train()
        
        masknum = 3
        # mask=np.zeros((self.args.CIL.base_class,self.args.CIL.num_classes))
        # for i in range(self.args.CIL.num_classes-self.args.CIL.base_class):
        #     picked_dummy=np.random.choice(self.args.CIL.base_class,masknum,replace=False)
        #     mask[:,i+self.args.CIL.base_class][picked_dummy]=1
        # mask=torch.tensor(mask).cuda()


        data = data.cuda()
        labels = label.cuda()


        logits = self._network(data)
        logits_ = logits[:, :self.base_class]
        # _, pred = torch.max(logits_, dim=1)
        # acc = self._accuracy(labels, pred)
        acc = accuracy(logits_, labels)[0]
        loss = self.loss(logits_, labels)
        loss.backward()
        
        ret = {}
        ret['loss'] = loss.item()
        ret['acc'] = acc.item()
        # print(ret)

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
        
        """
        base train for fact method
        Args:
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """
        self._network = trainer.model
        self._network.eval()
        
        masknum = 3
        # mask=np.zeros((self.args.CIL.base_class,self.args.CIL.num_classes))
        # for i in range(self.args.CIL.num_classes-self.args.CIL.base_class):
        #     picked_dummy=np.random.choice(self.args.CIL.base_class,masknum,replace=False)
        #     mask[:,i+self.args.CIL.base_class][picked_dummy]=1
        # mask=torch.tensor(mask).cuda()

        with torch.no_grad():
            data = data.cuda()
            labels = label.cuda()

        
            logits = self._network(data)
            logits_ = logits[:, :self.base_class]
            # _, pred = torch.max(logits_, dim=1)
            # acc = self._accuracy(labels, pred)
            acc = accuracy(logits_, labels)[0]
            loss = self.loss(logits_, labels)
            
            ret = {}
            ret['loss'] = loss.item()
            ret['acc'] = acc.item()

        return ret

    def test_step(self, trainer, data, label, *args, **kwargs):
        """Test step for standard supervised learning.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Test results. Contains the following:
                - "loss": Loss value.
                - "acc": Accuracy value.
        """
        return self.val_step(trainer, data, label, *args, **kwargs)

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