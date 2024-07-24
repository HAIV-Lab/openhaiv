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
    def __init__(self, trainer) -> None:
        super(FACT, self).__init__(trainer)
        self.args = trainer.cfg
        
        self.transform = None
        self.base_class = 11
        # self._network = FACTNET(self.args)
        self._network = None
        self.loss = nn.CrossEntropyLoss().cuda()
        self.beta = 0.5
        
        session = trainer.session
        if session==1:
            self._network = trainer.model
            self._network.eval()
            self._network.mode = self.args.CIL.new_mode
            print("network_fc: ",self._network.fc.weight.data[10])
            trainloader = trainer.train_loader
            tsfm = trainer.val_loader.dataset.transform
            trainloader.dataset.transform = tsfm
            print(np.unique(trainloader.dataset.targets))
            class_list = list(range(self.args.CIL.base_class+ (session-1)*self.args.CIL.way, self.args.CIL.base_class + self.args.CIL.way * session))
            self._network.update_fc(trainloader, class_list, session)
            print("network_fc: ",self._network.fc.weight.data[10])

    
    def train_step(self, trainer, data, label, *args, **kwargs):
        """
        base train for fact method
        Args:
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """
        session = trainer.session
        if session==0:
            self._network = trainer.model
            self._network.train()
            
            masknum = 3
            mask=np.zeros((self.args.CIL.base_class,self.args.CIL.num_classes))
            for i in range(self.args.CIL.num_classes-self.args.CIL.base_class):
                picked_dummy=np.random.choice(self.args.CIL.base_class,masknum,replace=False)
                mask[:,i+self.args.CIL.base_class][picked_dummy]=1
            mask=torch.tensor(mask).cuda()


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
        else:
            ret = {}
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
        session = trainer.session
        if session ==0:
            self._network = trainer.model
            self._network.eval()
            
            masknum = 3
            mask=np.zeros((self.args.CIL.base_class,self.args.CIL.num_classes))
            for i in range(self.args.CIL.num_classes-self.args.CIL.base_class):
                picked_dummy=np.random.choice(self.args.CIL.base_class,masknum,replace=False)
                mask[:,i+self.args.CIL.base_class][picked_dummy]=1
            mask=torch.tensor(mask).cuda()

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
        else: 
            test_class = self.args.CIL.base_class + session * self.args.CIL.way
            # self._network = trainer.model
            # self._network.eval()

            with torch.no_grad():
                data = data.cuda()
                labels = label.cuda()

                b = data.size()[0]
                # 20240711 
                if self.transform is not None:
                    data = self.transform(data)
                m = data.size()[0] // b
                joint_preds = self._network(data)
                feat = self._network.get_features(data)
                joint_preds = joint_preds[:, :test_class*m]
                
                agg_preds = 0
                agg_feat = feat.view(-1, m, feat.size(1)).mean(dim=1)
                for j in range(m):
                    agg_preds = agg_preds + joint_preds[j::m, j::m] / m

                acc = accuracy(agg_preds, labels)[0]
                # logits = self._network(data)
                # logits_ = logits[:, :self.args.CIL.base_class+self.args.CIL.base_class*session]
                # acc = accuracy(logits_, labels)[0]
                loss = self.loss(agg_preds, labels)
                
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


