import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from ncdia.utils import ALGORITHMS
from ncdia.algorithms.base import BaseAlg
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
from ncdia.dataloader import MergedDataset
from ncdia.utils.metrics import accuracy, per_class_accuracy
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import copy
from ncdia.models.net.alice_net import AliceNET
from ncdia.models.net.inc_net import IncrementalNet
from .fetril import filter_dataloader_by_class


@HOOKS.register
class SSREHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()

    def before_train(self, trainer) -> None:
        """
        在进入for epoch in range(max_epochs)循环之前，对训练数据集进行处理。
        """
        trainer._protos = []

    def after_train(self, trainer) -> None:
        """
        在进入for epoch in range(max_epochs)循环之后，对训练数据集进行处理。
        """
        print("after_train_epoch, build protos")
        trainer.algorithm._build_protos(trainer)

        old_model = IncrementalNet(
            trainer.cfg.model.network,
            trainer.cfg.CIL.base_classes,
            trainer.cfg.CIL.num_classes,
            trainer.cfg.CIL.att_classes,
            trainer.cfg.model.net_alice,
        )
        old_model.load_state_dict(trainer.model.state_dict())
        for param in old_model.parameters():
            param.requires_grad = False

        trainer.buffer["old_model"] = old_model


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


@ALGORITHMS.register
class SSRE(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.trainer = trainer

        hook = SSREHook()
        trainer.register_hook(hook)
        self._network = None
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        session = trainer.session
        self._network_module_ptr = None
        self.old_network_module_ptr = None

        self._protos = (
            []
            if session == 0
            else np.load(
                os.path.join("temp/", "protos.npy"), allow_pickle=True
            ).tolist()
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _build_protos(self, trainer):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                idx_dataset = self.trainer.test_loader.dataset
                target_classes = [class_idx]
                filtered_loader = filter_dataloader_by_class(
                    self.trainer.train_loader, self.trainer.test_loader, target_classes
                )

                vectors, _ = self._extract_vectors(filtered_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(class_mean)
            if not os.path.exists("temp/"):
                os.makedirs("temp/")
            np.save("temp/protos.npy", self._protos)
            print("build protos done")

    def _compute_ssre_loss(self, inputs, targets, session):
        if session == 0:
            logits = self._network(inputs)
            logits_ = logits[:, : self._total_classes]
            # loss_clf = F.cross_entropy(logits/self.args["temp"], targets)
            loss_clf = F.cross_entropy(logits_, targets)
            return logits, loss_clf, torch.tensor(0.0), torch.tensor(0.0)

        features = self._network_module_ptr.extract_vector(inputs)  # N D

        with torch.no_grad():
            features_old = self.old_network_module_ptr.extract_vector(inputs)

        protos = torch.from_numpy(np.array(self._protos)).to(self._device)
        protos = protos.float()
        with torch.no_grad():
            weights = (
                F.normalize(features, p=2, dim=1, eps=1e-12)
                @ F.normalize(protos, p=2, dim=1, eps=1e-12).T
            )
            weights = torch.max(weights, dim=1)[0]
            # mask = weights > self.args["threshold"]
            mask = weights
        logits = self._network(inputs)
        # loss_clf = F.cross_entropy(logits/self.args["temp"],targets,reduction="none")
        logits_ = logits[:, : self._total_classes]
        loss_clf = F.cross_entropy(logits_, targets, reduction="none")
        # loss_clf = F.cross_entropy(logits,targets,reduction="none")
        # loss_clf = torch.mean(loss_clf * ~mask)
        loss_clf = torch.mean(loss_clf * (1 - mask))

        loss_fkd = torch.norm(features - features_old, p=2, dim=1)
        loss_fkd = self.args.CIL.lambda_fkd * torch.sum(loss_fkd * mask)

        known_class = max(
            self.args.CIL.base_classes + (session - 1) * self.args.CIL.way, 0
        )
        index = np.random.choice(
            range(known_class), size=self.trainer.train_loader.batch_size, replace=True
        )

        proto_features = np.array(self._protos)[index]
        proto_targets = index
        proto_features = proto_features
        proto_features = (
            torch.from_numpy(proto_features).float().to(self._device, non_blocking=True)
        )
        proto_targets = torch.from_numpy(proto_targets).to(
            self._device, non_blocking=True
        )

        proto_logits = self._network_module_ptr.fc(proto_features)
        proto_logits = proto_logits[:, : self._total_classes]
        loss_proto = self.args.CIL.lambda_proto * F.cross_entropy(
            proto_logits, proto_targets
        )
        return logits, loss_clf, loss_fkd, loss_proto

    def train_step(self, trainer, data, label, attribute, imgpath):
        """
        base train for fact method
        Args:
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """
        session = trainer.session
        self._network = trainer.model
        if session >= 1:
            self._old_network = trainer.buffer["old_model"]
            self._old_network = self._old_network.cuda()
            self._old_network.eval()
            if hasattr(self._old_network, "module"):
                self.old_network_module_ptr = self._old_network.module
            else:
                self.old_network_module_ptr = self._old_network
            self.old_network_module_ptr.cuda()
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.cuda()
        self._network_module_ptr.train()

        known_class = max(
            self.args.CIL.base_classes + (session - 1) * self.args.CIL.way, 0
        )
        self._known_classes = known_class
        self._total_classes = self.args.CIL.base_classes + (session) * self.args.CIL.way

        self._network = trainer.model
        self._network.train()

        data = data.cuda()
        labels = label.cuda()
        logits, loss_clf, loss_fkd, loss_proto = self._compute_ssre_loss(
            data, labels, session
        )
        logits_ = logits[:, : self._total_classes]
        acc = accuracy(logits_, labels)[0]
        per_acc = str(per_class_accuracy(logits_, labels))
        loss = loss_clf + loss_fkd + loss_proto
        loss.backward()

        ret = {}
        ret["loss"] = loss
        ret["acc"] = acc
        ret["per_class_acc"] = per_acc

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

        ret = {}
        ret["loss"] = loss.item()
        ret["acc"] = acc.item()
        ret["per_class_acc"] = per_acc

        return ret

    def test_step(self, trainer, data, label, *args, **kwargs):
        return self.val_step(trainer, data, label, *args, **kwargs)

    def get_net(self):
        return self._network
