import os
import logging
import itertools
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ncdia.utils import ALGORITHMS
from ncdia.algorithms.base import BaseAlg
from ncdia.utils.losses import AngularPenaltySMLoss
from ncdia.utils.metrics import accuracy, per_class_accuracy
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import QuantifyHook
from ncdia.models.net.inc_net import IncrementalNet
from ncdia.dataloader import MergedDataset
from ncdia.dataloader import BaseDataset


@HOOKS.register
class iCaRLHook(QuantifyHook):
    def __init__(self) -> None:
        super().__init__()
        self._fix_memory = True

    def before_train(self, trainer) -> None:
        trainer.train_loader
        _hist_trainset = trainer.hist_trainset
        now_dataset = trainer.train_loader.dataset
        _hist_trainset = MergedDataset([_hist_trainset], replace_transform=True)

        if self._fix_memory and trainer.session >= 1:
            _hist_trainset, new_dataset = self.construct_exemplar_unified(
                _hist_trainset, trainer.cfg.CIL.per_classes, trainer
            )
            _hist_trainset = MergedDataset([_hist_trainset], replace_transform=True)
            new_dataset = MergedDataset([new_dataset], replace_transform=True)

        if trainer.session >= 1:
            _hist_trainset.merge([new_dataset], replace_transform=True)
        else:
            _hist_trainset.merge([now_dataset], replace_transform=True)
        # print(_hist_trainset.labels)
        trainer._train_loader = DataLoader(
            _hist_trainset, **trainer._train_loader_kwargs
        )

        # val_loader
        trainer.val_loader
        _hist_valset = MergedDataset([trainer.hist_valset], replace_transform=True)
        _hist_valset.merge([trainer.val_loader.dataset], replace_transform=True)
        trainer._val_loader = DataLoader(_hist_valset, **trainer._val_loader_kwargs)

    def after_train(self, trainer) -> None:
        trainer.update_hist_dataset(
            key="hist_trainset",
            new_dataset=trainer.train_loader.dataset,
            replace_transform=True,
            inplace=True,
        )

        trainer.update_hist_dataset(
            key="hist_valset",
            new_dataset=trainer.val_loader.dataset,
            replace_transform=True,
            inplace=True,
        )

        algorithm = trainer.algorithm
        filename = "task_" + str(trainer.session) + ".pth"
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))
        old_model = IncrementalNet(
            trainer.cfg.model.network,
            trainer.cfg.CIL.base_classes,
            trainer.cfg.CIL.num_classes,
            trainer.cfg.model.net_alice,
        )
        old_model.load_state_dict(trainer.model.state_dict())
        for param in old_model.parameters():
            param.requires_grad = False

        trainer.buffer["old_model"] = old_model

    def construct_exemplar_unified(self, trainset, m, trainer):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )

        trainset.merge([trainer.train_loader.dataset], replace_transform=True)
        args = trainer.cfg
        session = trainer.session
        total_class = args.CIL.base_classes + (session) * args.CIL.way
        known_class = max(args.CIL.base_classes + (session - 1) * args.CIL.way, 0)
        start_class = max(args.CIL.base_classes + (session - 2) * args.CIL.way, 0)
        _network = trainer.model
        _feature_dim = 2048
        class_means = np.zeros((total_class, _feature_dim))

        data_loader = DataLoader(trainset, **trainer._train_loader_kwargs)
        class_sums = np.zeros((total_class, _feature_dim))
        class_counts = np.zeros(total_class)

        for i, batch in enumerate(tqdm(data_loader, desc="Calculating class means")):
            images = batch["data"]
            labels = batch["label"]
            images = images.cuda()
            with torch.no_grad():
                features = _network.extract_vector(images)
            features_np = features.cpu().numpy()

            for i in range(len(labels)):
                class_index = labels[i].item()
                class_sums[class_index] += features_np[i]
                class_counts[class_index] += 1

        for i in range(total_class):
            if class_counts[i] > 0:
                mean = class_sums[i] / class_counts[i]
                class_means[i] = mean

        selected_indices = {i: [] for i in range(known_class, total_class)}
        selected_features = {i: [] for i in range(known_class, total_class)}

        all_features = []
        all_labels = []
        all_imgpaths = []

        for batch in tqdm(data_loader, desc="Gathering all samples"):
            images = batch["data"].cuda()
            labels = batch["label"].cpu().numpy()
            imgpaths = batch["imgpath"]

            images = images.contiguous().float()

            with torch.no_grad():
                features = _network.extract_vector(images).cpu().numpy()

            all_features.append(features)
            all_labels.append(labels)
            all_imgpaths.append(imgpaths)

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_imgpaths = list(itertools.chain.from_iterable(all_imgpaths))
        # all_imgpaths = np.concatenate(all_imgpaths, axis=0)
        # print(all_imgpaths)

        for class_id in tqdm(
            range(start_class, known_class), desc="Selecting nearest samples"
        ):
            class_indices = np.where(all_labels == class_id)[0]
            if len(class_indices) == 0:
                continue

            class_center = class_means[class_id]

            distances = np.linalg.norm(
                all_features[class_indices] - class_center, axis=1
            )

            nearest_indices = np.argsort(distances)[:m]
            selected_indices[class_id] = class_indices[nearest_indices].tolist()
            selected_features[class_id] = all_features[class_indices][nearest_indices]

        retained_images = []
        retained_labels = []
        for class_id in range(start_class, known_class):
            indices = selected_indices[class_id]
            retained_images.extend([all_imgpaths[i] for i in indices])
            retained_labels.extend([all_labels[i] for i in indices])

        retained_datasets = BaseDataset(
            loader=trainer.train_loader.dataset.loader,
            transform=trainer.train_loader.dataset.transform,
        )

        retained_datasets.images = retained_images
        retained_datasets.labels = retained_labels

        all_remaining_images = []
        all_remaining_labels = []
        for class_id in range(known_class, total_class):
            class_indices = np.where(all_labels == class_id)[0]
            all_remaining_images.extend([all_imgpaths[i] for i in class_indices])
            all_remaining_labels.extend([all_labels[i] for i in class_indices])
        all_remaining_datasets = BaseDataset(
            loader=trainer.train_loader.dataset.loader,
            transform=trainer.train_loader.dataset.transform,
        )
        all_remaining_datasets.images = all_remaining_images
        all_remaining_datasets.labels = all_remaining_labels

        return retained_datasets, all_remaining_datasets

    def before_test(self, trainer) -> None:
        trainer.test_loader
        _hist_testset = MergedDataset([trainer.hist_testset], replace_transform=True)
        _hist_testset.merge([trainer.test_loader.dataset], replace_transform=True)
        trainer._test_loader = DataLoader(_hist_testset, **trainer._test_loader_kwargs)

    def after_test(self, trainer) -> None:

        trainer.update_hist_dataset(
            key="hist_testset",
            new_dataset=trainer.test_loader.dataset,
            replace_transform=True,
            inplace=True,
        )


@ALGORITHMS.register
class iCaRL(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.args = trainer.cfg
        self.trainer = trainer

        self._network = None
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        self.hook = iCaRLHook()
        trainer.register_hook(self.hook)

        session = trainer.session

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
        known_class = self.args.CIL.base_classes + session * self.args.CIL.way
        self._network = trainer.model
        if session >= 1:
            self._old_network = trainer.buffer["old_model"]
            self._old_network = self._old_network.cuda()
            self._old_network.eval()

        self._network.train()

        data = data.cuda()
        labels = label.cuda()
        logits = self._network(data)
        if session >= 1:
            with torch.no_grad():
                old_logits = self._old_network(data)
        logits_ = logits[:, :known_class]
        acc = accuracy(logits_, labels)[0]
        per_acc = str(per_class_accuracy(logits_, labels))
        loss = self.loss(logits_, labels)
        if session >= 1:
            kd_loss = self._KD_loss(logits_, old_logits[:, :known_class], 1.0)
            loss = loss + 3.0 * kd_loss
        loss.backward()

        ret = {}
        ret["loss"] = loss
        ret["acc"] = acc
        # ret['per_class_acc'] = per_acc
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

    def _KD_loss(self, pred, soft, T):
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
