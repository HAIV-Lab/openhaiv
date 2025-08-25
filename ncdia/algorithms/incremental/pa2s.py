import torch
import numpy as np

from ncdia.utils import ALGORITHMS
from ncdia.algorithms.base import BaseAlg
from ncdia.utils.losses import AngularPenaltySMLoss
from ncdia.utils.metrics import accuracy, per_class_accuracy
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
from ncdia.models.net.inc_net import IncrementalNet

import os
import logging
from tqdm import tqdm
import itertools
from torch import optim
import copy
from torch.utils.data import DataLoader
from ncdia.dataloader import MergedDataset
from ncdia.dataloader import BaseDataset


@HOOKS.register
class PASSHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
        self.radii = np.array([])
        self.prototype = np.array([])
        self.class_label = np.array([])

    def before_train(self, trainer) -> None:
        if "prototype" not in trainer.buffer:
            trainer.buffer["prototype"] = self.prototype
        if "class_label" not in trainer.buffer:
            trainer.buffer["class_label"] = self.class_label
        if "radii" not in trainer.buffer:
            trainer.buffer["radii"] = self.radii
        if "radius" not in trainer.buffer:
            trainer.buffer["radius"] = 0

    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        filename = "task_" + str(trainer.session) + ".pth"
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))
        trainer.buffer["old_model"] = IncrementalNet(
            trainer.cfg.model.network,
            trainer.cfg.CIL.base_classes,
            trainer.cfg.CIL.num_classes,
            trainer.cfg.model.net_alice,
        )
        trainer.buffer["old_model"].load_state_dict(trainer.model.state_dict())
        for param in trainer.buffer["old_model"].parameters():
            param.requires_grad = False

        self._build_protos(trainer)

        if trainer.session == 0:
            trainer.buffer["prototype"] = self.prototype
            trainer.buffer["class_label"] = self.class_label
            trainer.buffer["radii"] = self.radii
        else:
            trainer.buffer["prototype"] = np.concatenate(
                (trainer.buffer["prototype"], self.prototype), axis=0
            )
            trainer.buffer["class_label"] = np.concatenate(
                (trainer.buffer["class_label"], self.class_label), axis=0
            )
            trainer.buffer["radii"] = np.concatenate(
                (trainer.buffer["radii"], self.radii), axis=0
            )
        trainer.buffer["radius"] = np.sqrt(np.mean(trainer.buffer["radii"]))

    def _build_protos(self, trainer):
        session = trainer.session
        _network = trainer.model
        _loader = trainer.train_loader
        features = []
        labels = []
        radii = []
        prototype = []

        if session == 0:
            known_class = 0
        else:
            known_class = (
                trainer.cfg.CIL.base_classes + (session - 1) * trainer.cfg.CIL.way
            )
        total_class = trainer.cfg.CIL.base_classes + session * trainer.cfg.CIL.way

        _network.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(_loader, desc="Building prototypes")):
                images = batch["data"].cuda()
                features.append(_network.extract_vector(images).cpu().numpy())
                labels.append(batch["label"].numpy())

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        class_set = range(known_class, total_class)
        feature_dim = features.shape[-1]

        for cls in class_set:
            index = np.where(cls == labels)[0]
            feature_cls = features[index]
            prototype.append(np.mean(feature_cls, axis=0))
            cov = np.cov(feature_cls.T)
            radii.append(np.trace(cov) / feature_dim)

        self.prototype = np.array(prototype)
        self.radii = np.array(radii)
        self.class_label = class_set


@ALGORITHMS.register
class PASS(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.args = trainer.cfg
        self.trainer = trainer

        self._network = None
        self._old_network = None
        self.transform = None
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        self.hook = PASSHook()
        trainer.register_hook(self.hook)

        self.batchsize = 64
        self.temperature = 0.1
        self.kd_weight = 10
        self.proto_weight = 10

    def train_step(self, trainer, data, label, attribute, imgpath):
        """
        base train for PASS method
        Args:
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """

        session = self.trainer.session
        self._network = trainer.model
        if session >= 1:
            self._old_network = trainer.buffer["old_model"]
            self._old_network = self._old_network.cuda()
            self._old_network.eval()

        self._network.train()

        data = data.cuda()
        labels = label.cuda()
        loss, acc, per_acc = self._compute_loss(data, labels)
        loss.backward()

        ret = {"loss": loss, "acc": acc, "per_class_acc": per_acc}

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
        test_class = self.args.CIL.base_classes + session * self.args.CIL.way
        self._network = trainer.model
        self._network.eval()
        data = data.cuda()
        labels = label.cuda()
        logits = self._network(data)
        logits_ = logits[:, :test_class]
        acc = accuracy(logits_, labels)[0]
        loss = self.loss(logits_, labels)
        per_acc = str(per_class_accuracy(logits_, labels))

        ret = {"loss": loss.item(), "acc": acc.item(), "per_class_acc": per_acc}

        return ret

    def test_step(self, trainer, data, label, *args, **kwargs):
        return self.val_step(trainer, data, label, *args, **kwargs)

    def get_net(self):
        return self._network

    def _compute_loss(self, data, labels):
        session = self.trainer.session
        if session == 0:
            known_class = self.args.CIL.base_classe
        else:
            known_class = self.args.CIL.base_classes + (session - 1) * self.args.CIL.way
        total_class = self.args.CIL.base_classes + session * self.args.CIL.way
        logits = self._network(data)
        logits_ = logits[:, :total_class]
        acc = accuracy(logits_, labels)[0]
        per_acc = str(per_class_accuracy(logits_, labels))

        loss_cls = self.loss(logits_ / self.temperature, labels)
        if self._old_network is None:
            loss = loss_cls
        else:
            feature = self._network.extract_vector(data)
            feature_old = self._old_network.extract_vector(data)
            loss_kd = self.kd_weight * torch.dist(feature, feature_old, 2)

            index = np.random.choice(
                range(known_class),
                size=self.batchsize
                * int(np.ceil(known_class / (total_class - known_class))),
                replace=True,
            )
            proto_features = self.trainer.buffer["prototype"][index]
            proto_labels = index
            proto_features = (
                proto_features
                + np.random.normal(0, 1, proto_features.shape)
                * self.trainer.buffer["radius"]
            )
            proto_features = torch.from_numpy(proto_features).float().cuda()
            proto_labels = torch.from_numpy(proto_labels).cuda()
            proto_logits = self._network.fc(proto_features)
            loss_proto = self.proto_weight * self.loss(
                proto_logits / self.temperature, proto_labels
            )

            loss = loss_cls + loss_kd + loss_proto

        return loss, acc, per_acc
