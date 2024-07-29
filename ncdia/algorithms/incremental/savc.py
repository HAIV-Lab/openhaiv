import torch
import torch.nn.functional as F

from ncdia.utils import ALGORITHMS
from ncdia.algorithms.base import BaseAlg
from .losses.angular_loss import AngularPenaltySMLoss
from . import fantasy


@ALGORITHMS.register
class SAVC(BaseAlg):
    def __init__(self, trainer) -> None:
        print("++++++++++++++++++++++++++cfg:")
        super().__init__()
        self.trainer = trainer
        self.args = trainer.cfg
        self.config = trainer.cfg

        self._network = None
        self.loss = AngularPenaltySMLoss()

        if self.args.CIL.fantasy is not None:
            self.transform, self.num_trans = fantasy.__dict__[self.args.CIL.fantasy]()
        else:
            self.transform = None
            self.num_trans = 0
        print("++++++++++++++++++++++++++++++++++++++")

    def train_step(self, trainer, data, label, attribute, imgpath):
        """
        base train for fact method
        Args:
            trainer: trainer object
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """
        self._network = trainer.model
        self._network.train()
        device = trainer.device

        b, c, h, w = data[1].shape
        original = data[0].to(device)
        data[1] = data[1].to(device)
        data[2] = data[2].to(device)
        label = label.to(device)

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
        loss.backward()


        acc = self._accuracy(joint_labels, joint_preds)

        ret = {}
        ret['loss'] = loss
        ret['acc'] = acc

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
        self._network.eval()
        device = trainer.device
        logit_list = [], pred_list = [], conf_list = [], label_list = []
        feature_list = []
        test_class = self.config.CIL.base_class + 1 * self.config.CIL.way
        with torch.no_grad:
            data = data.to(device)
            label = label.to(device)
            b = data.size()[0]
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
            
            loss = F.cross_entropy(joint_preds, label)
            
            feature_list.append(agg_feat)
            logit_list.append(agg_preds)
            score = torch.softmax(agg_preds, dim=1)
            conf, pred = torch.max(score, dim=1)
            acc = self._accuracy(pred, label)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        feature_list = torch.cat(feature_list, dim=0).cpu().numpy()
        logit_list = torch.cat(logit_list, dim=0).numpy()
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        ret = {
            "feature": feature_list,
            "logits": logit_list,
            "predicts": pred_list,
            "confidence":conf_list,
            "labels": label_list,
            "acc": acc,
            "loss": loss
        }
        
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