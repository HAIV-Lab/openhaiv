import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ncdia.algorithms.base import BaseAlg
from ncdia.utils import ALGORITHMS
from ncdia.utils.metrics import accuracy
from ncdia.trainers import PreTrainer
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
from ncdia.trainers.hooks import QuantifyHook
from ncdia.models.net.inc_net import IncrementalNet
from ncdia.dataloader import MergedDataset
from ncdia.dataloader import BaseDataset


class FACTHook(QuantifyHook):
    def __init__(self) -> None:
        super().__init__()

    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        algorithm.replace_fc()
        filename = "task_" + str(trainer.session) + ".pth"
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))
        if trainer.session == 0:
            self.save_train_static(trainer)

    def save_train_static(self, trainer):
        all_class = trainer.train_loader.dataset.num_classes
        features, logits, labels = [], [], []
        tbar = tqdm(trainer.train_loader, dynamic_ncols=True, disable=True)
        for batch in tbar:
            data = batch["data"].to(trainer.device)
            label = batch["label"].to(trainer.device)
            joint_preds = trainer.model(data)
            joint_preds = joint_preds[:, :all_class]
            feats = trainer.model.get_features(data)

            features.append(feats.clone().detach().cpu())
            logits.append(joint_preds.clone().detach().cpu())
            labels.append(label.clone().detach().cpu())

        features = torch.cat(features, dim=0)
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0).to(torch.int)

        classes = torch.unique(labels)
        prototype_cls = []
        for cls in classes:
            cls_indices = torch.where(labels == cls)
            cls_preds = logits[cls_indices]
            prototype_cls.append(torch.mean(cls_preds, dim=0))
        filename = "train_static.pt"
        torch.save(
            {
                "train_features": features,
                "train_logits": logits,
                "prototype": torch.stack(prototype_cls),
            },
            os.path.join(trainer.work_dir, filename),
        )


@ALGORITHMS.register
class FACT(BaseAlg):
    def __init__(self, trainer) -> None:
        super(FACT, self).__init__(trainer)
        self.args = trainer.cfg
        self.trainer = trainer

        self.transform = None
        self._network = None
        self.loss = nn.CrossEntropyLoss().cuda()
        self.beta = 0.5
        self.hook = FACTHook()
        trainer.register_hook(self.hook)

        session = trainer.session
        if session >= 1:
            self._network = trainer.model
            self._network.eval()
            self._network.mode = self.args.CIL.new_mode
            trainloader = trainer.train_loader
            tsfm = trainer.val_loader.dataset.transform
            trainloader.dataset.transform = tsfm
            class_list = list(
                range(
                    self.args.CIL.base_class + (session - 1) * self.args.CIL.way,
                    self.args.CIL.base_class + self.args.CIL.way * session,
                )
            )
            self._network.update_fc(trainloader, class_list, session)
            # print("network_fc: ",self._network.fc.weight.data[12])

    def replace_fc(self):
        session = self.trainer.session
        if not self.args.CIL.not_data_init and session == 0:
            train_loader = self.trainer.train_loader
            val_loader = self.trainer.val_loader
            train_loader.dataset.multi_train = False
            train_loader.dataset.transform = val_loader.dataset.transform
            self._network = self.trainer.model
            self._network.eval()
            embedding_list = []
            label_list = []
            with torch.no_grad():
                for i, batch in enumerate(train_loader):
                    data = batch["data"].cuda()
                    label = batch["label"].cuda()

                    b = data.size()[0]
                    m = data.size()[0] // b
                    labels = torch.stack([label * m + ii for ii in range(m)], 1).view(
                        -1
                    )
                    embedding = self._network.get_features(data)

                    embedding_list.append(embedding.cpu())
                    label_list.append(labels.cpu())
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

            proto_list = []
            for class_index in range(self.args.CIL.base_class * m):
                data_index = (label_list == class_index).nonzero()
                embedding_this = embedding_list[data_index.squeeze(-1)]
                embedding_this = embedding_this.mean(0)
                proto_list.append(embedding_this)

            proto_list = torch.stack(proto_list, dim=0)

            self._network.fc.weight.data[: self.args.CIL.base_class * m] = proto_list

            # return self.net
            # class_list = list(range(self.args.CIL.base_class))
            # print(class_list)
            # self._network.update_fc(train_loader, class_list, 0)

    def train_step(self, trainer, data, label, *args, **kwargs):
        """
        base train for fact method
        Args:
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """
        session = self.trainer.session
        if session == 0:
            self._network = trainer.model
            self._network.train()

            masknum = 3
            mask = np.zeros((self.args.CIL.base_class, self.args.CIL.num_classes))
            for i in range(self.args.CIL.num_classes - self.args.CIL.base_class):
                picked_dummy = np.random.choice(
                    self.args.CIL.base_class, masknum, replace=False
                )
                mask[:, i + self.args.CIL.base_class][picked_dummy] = 1
            mask = torch.tensor(mask).cuda()

            data = data.cuda()
            labels = label.cuda()

            logits = self._network(data)
            logits_ = logits[:, : self.args.CIL.base_class]
            # _, pred = torch.max(logits_, dim=1)
            # acc = self._accuracy(labels, pred)
            acc = accuracy(logits_, labels)[0]
            loss = self.loss(logits_, labels)
            loss.backward()

            ret = {}
            ret["loss"] = loss.item()
            ret["acc"] = acc.item()
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
        session = self.trainer.session
        if session == 0:
            self._network = trainer.model
            self._network.eval()

            masknum = 3
            mask = np.zeros((self.args.CIL.base_class, self.args.CIL.num_classes))
            for i in range(self.args.CIL.num_classes - self.args.CIL.base_class):
                picked_dummy = np.random.choice(
                    self.args.CIL.base_class, masknum, replace=False
                )
                mask[:, i + self.args.CIL.base_class][picked_dummy] = 1
            mask = torch.tensor(mask).cuda()

            with torch.no_grad():
                data = data.cuda()
                labels = label.cuda()

                logits = self._network(data)
                logits_ = logits[:, : self.args.CIL.base_class]
                # _, pred = torch.max(logits_, dim=1)
                # acc = self._accuracy(labels, pred)
                acc = accuracy(logits_, labels)[0]
                loss = self.loss(logits_, labels)

                ret = {}
                ret["loss"] = loss.item()
                ret["acc"] = acc.item()
        else:
            test_classes = self.args.CIL.base_class + session * self.args.CIL.way
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
                joint_preds = joint_preds[:, : test_classes * m]

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
                ret["loss"] = loss.item()
                ret["acc"] = acc.item()
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
