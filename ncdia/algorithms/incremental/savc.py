import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm


from ncdia.utils.cfg import Configs
from ncdia.utils.logger import Logger
from ncdia.utils import INMETHODS
from .base import BaseLearner
from .net.savc_net import SAVCNET
from .losses.angular_loss import AngularPenaltySMLoss
import fantasy


@INMETHODS.register()
class SAVC(BaseLearner):
    def __init__(self, cfg: Configs) -> None:
        self.args = cfg.copy()
        super().__init__(self.args)

        self._network = SAVCNET(self.args)
        self.loss = AngularPenaltySMLoss()

        if self.args.CIL.fantasy is not None:
            self.transform, self.num_trans = fantasy.__dict__[self.args.CIL.fantasy]()
        else:
            self.transform = None
            self.num_trans = 0
    
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

        loss_avg = 0.0
        b, c, h, w = data[1].shape
        original = data[0].cuda()
        data[1] = data[1].cuda()
        data[2] = data[2].cuda()
        label = label.cuda()

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
        joint_labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)

        # ------  forward  ------- #
        joint_preds = self.net(im_cla=data_classify)  
        joint_preds = joint_preds[:, :self.config.CIL.base_class*m]
        joint_loss = F.cross_entropy(joint_preds, joint_labels)

        agg_preds = 0
        for i in range(m):
            agg_preds = agg_preds + joint_preds[i::m, i::m] / m

        loss = joint_loss
        acc = self._accuracy(joint_labels, joint_preds)

        ret = {}
        ret['loss'] = loss
        ret['acc'] = acc

        return ret



    def _incremental_train(self):
        return self._network

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