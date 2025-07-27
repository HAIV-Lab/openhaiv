import os
import torch
import numpy as np

from ncdia.utils import ALGORITHMS
from ncdia.algorithms.base import BaseAlg
from ncdia.utils.losses import AngularPenaltySMLoss
from ncdia.utils.metrics import accuracy, per_class_accuracy
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
from ncdia.trainers.hooks import QuantifyHook
from ncdia.models.net.inc_net import IncrementalNet
from ncdia.dataloader import MergedDataset
from ncdia.dataloader import BaseDataset
from torch.utils.data import DataLoader


@HOOKS.register
class AliceHook(QuantifyHook):
    def __init__(self) -> None:
        super().__init__()

    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        algorithm.replace_fc()

        filename = "task_" + str(trainer.session) + ".pth"
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))
        if trainer.session == 0:
            self.gather_stats(trainer.model, trainer.train_loader, trainer.device)

    def before_test(self, trainer) -> None:
        """
        在进入for epoch in range(max_epochs)循环之前，对测试数据集进行处理。
        """
        trainer.test_loader
        _hist_testset = MergedDataset([trainer.hist_testset], replace_transform=True)
        _hist_testset.merge([trainer.test_loader.dataset], replace_transform=True)
        trainer._test_loader = DataLoader(_hist_testset, **trainer._test_loader_kwargs)

    def after_test(self, trainer) -> None:
        """ """
        trainer.update_hist_dataset(
            key="hist_valset",
            new_dataset=trainer.val_loader.dataset,
            replace_transform=True,
            inplace=True,
        )


@ALGORITHMS.register
class Alice(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.args = trainer.cfg
        self.trainer = trainer

        # self._network = AliceNET(self.args)
        self._network = None
        self.transform = None
        self.loss = AngularPenaltySMLoss(loss_type="cosface").cuda()
        # self.loss = nn.CrossEntropyLoss().cuda()
        self.hook = AliceHook()
        trainer.register_hook(self.hook)

        session = trainer.session
        if session >= 1:
            self._network = trainer.model
            self._network.eval()
            self._network.cuda()
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
            # proto_list = torch.nn.functional.normalize(proto_list, p=2, dim=0)
            self._network.fc.weight.data[: self.args.CIL.base_class * m] = proto_list

            # return self.net
            # class_list = list(range(self.args.CIL.base_class))
            # print(class_list)
            # self._network.update_fc(train_loader, class_list, 0)

    def train_step(self, trainer, data, label, attribute, imgpath):
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
            # pred = F.softmax(logits_, dim=1)
            acc = accuracy(logits_, labels)[0]
            per_acc = str(per_class_accuracy(logits_, labels))
            loss = self.loss(logits_, labels)
            loss.backward()

            ret = {}
            ret["loss"] = loss
            ret["acc"] = acc
            ret["per_class_acc"] = per_acc
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
        session = self.trainer.session

        if session == 0:
            self._network = trainer.model
            self._network.eval()

            with torch.no_grad():
                data = data.cuda()
                labels = label.cuda()

                logits = self._network(data)
                logits_ = logits[:, : self.args.CIL.base_class]
                # _, pred = torch.max(logits_, dim=1)
                acc = accuracy(logits_, labels)[0]
                loss = self.loss(logits_, labels)
                per_acc = str(per_class_accuracy(logits_, labels))

                ret = {}
                ret["loss"] = loss.item()
                ret["acc"] = acc.item()
                ret["per_class_acc"] = per_acc
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
                joint_preds = joint_preds[:, : test_class * m]

                agg_preds = 0
                agg_feat = feat.view(-1, m, feat.size(1)).mean(dim=1)
                for j in range(m):
                    agg_preds = agg_preds + joint_preds[j::m, j::m] / m

                acc = accuracy(agg_preds, labels)[0]
                # logits = self._network(data)
                # logits_ = logits[:, :self.args.CIL.base_class+self.args.CIL.base_class*session]
                # acc = accuracy(logits_, labels)[0]
                loss = self.loss(agg_preds, labels)
                per_acc = str(per_class_accuracy(agg_preds, labels))

                ret = {}
                ret["loss"] = loss.item()
                ret["acc"] = acc.item()
                ret["per_class_acc"] = per_acc

        return ret

    def test_step(self, trainer, data, label, *args, **kwargs):
        return self.val_step(trainer, data, label, *args, **kwargs)

    def _incremental_train(self):
        pass

    def get_net(self):
        return self._network
