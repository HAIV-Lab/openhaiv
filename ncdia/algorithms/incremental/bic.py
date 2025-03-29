import os
import logging
<<<<<<< HEAD
import itertools
import numpy as np 
=======
import numpy as np
>>>>>>> 266289a9334291351ac9ac88159866e6b72faf8d
from tqdm import tqdm
import itertools
from torch import optim
import copy

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
from ncdia.trainers.hooks import AlgHook
from ncdia.models.net.inc_net import IncrementalNet

from torch.utils.data import DataLoader
from ncdia.dataloader import MergedDataset
from ncdia.dataloader import BaseDataset

from ncdia.models.net.inc_net import IncrementalNet

@HOOKS.register
class BiCHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
        self._fix_memory = True

    def before_train(self, trainer) -> None:
        _ = trainer.train_loader
        _hist_trainset = trainer.hist_trainset
        now_dataset = trainer.train_loader.dataset
        _hist_trainset = MergedDataset([_hist_trainset], replace_transform=True)
        # trainer._train_loader = DataLoader(_hist_trainset, **trainer._train_loader_kwargs)

        if self._fix_memory and trainer.session > 0:
            _hist_trainset , new_dataset = self.construct_exemplar_unified(_hist_trainset, trainer.cfg.CIL.per_classes, trainer)
            _hist_trainset = MergedDataset([_hist_trainset], replace_transform=True)
            new_dataset = MergedDataset([new_dataset], replace_transform=True)

        if trainer.session > 0:
            _hist_trainset.merge([new_dataset], replace_transform=True)
        else:
            _hist_trainset.merge([now_dataset], replace_transform=True)
        trainer._train_loader = DataLoader(_hist_trainset, **trainer._train_loader_kwargs)

        _ = trainer.val_loader
        _hist_valset = MergedDataset([trainer.hist_valset], replace_transform=True)
        _hist_valset.merge([trainer.val_loader.dataset], replace_transform=True)
        trainer._val_loader = DataLoader(_hist_valset, **trainer._val_loader_kwargs)

    def after_train(self, trainer) -> None:
        trainer.update_hist_dataset(
<<<<<<< HEAD
            key = 'hist_trainset',
            new_dataset = trainer.train_loader.dataset,
=======
            'hist_trainset',
            trainer.train_loader.dataset,
>>>>>>> 266289a9334291351ac9ac88159866e6b72faf8d
            replace_transform=True,
            inplace=True
        )

        trainer.update_hist_dataset(
<<<<<<< HEAD
            key = 'hist_valset',
            new_dataset = trainer.val_loader.dataset,
=======
            'hist_valset',
            trainer.val_loader.dataset,
>>>>>>> 266289a9334291351ac9ac88159866e6b72faf8d
            replace_transform=True,
            inplace=True
        )
        algorithm = trainer.algorithm
        filename = 'task_' + str(trainer.session) + '.pth'
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))
<<<<<<< HEAD
        old_model = IncrementalNet(
=======
        trainer.buffer["old_model"] = IncrementalNet(
>>>>>>> 266289a9334291351ac9ac88159866e6b72faf8d
            trainer.cfg.model.network,
            trainer.cfg.CIL.base_classes,
            trainer.cfg.CIL.num_classes,
            trainer.cfg.CIL.att_classes,
            trainer.cfg.model.net_alice
        )
<<<<<<< HEAD
        old_model.load_state_dict(trainer.model.state_dict())
        for param in old_model.parameters():
=======
        trainer.buffer["old_model"].load_state_dict(trainer.model.state_dict())
        for param in trainer.buffer["old_model"].parameters():
>>>>>>> 266289a9334291351ac9ac88159866e6b72faf8d
            param.requires_grad = False
        trainer.buffer['old_model'] = old_model

    def before_val(self, trainer) -> None:
        pass

    def after_val(self, trainer) -> None:
        pass


    def before_test(self, trainer) -> None:
        _ = trainer.test_loader
        _hist_testset = MergedDataset([trainer.hist_testset], replace_transform=True)
        _hist_testset.merge([trainer.test_loader.dataset], replace_transform=True)
        trainer._test_loader = DataLoader(_hist_testset, **trainer._test_loader_kwargs)

    def after_test(self, trainer) -> None:
        trainer.update_hist_dataset(
<<<<<<< HEAD
            key = 'hist_testset',
            new_dataset = trainer.test_loader.dataset,
=======
            'hist_testset',
            trainer.test_loader.dataset,
>>>>>>> 266289a9334291351ac9ac88159866e6b72faf8d
            replace_transform=True,
            inplace=True
        )

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
        _feature_dim = 512
        class_means = np.zeros((total_class, _feature_dim))

        data_loader = DataLoader(trainset, **trainer._train_loader_kwargs)
        class_sums = np.zeros((total_class, _feature_dim))
        class_counts = np.zeros(total_class)


        for i, batch in enumerate(tqdm(data_loader, desc="Calculating class means")):
            images = batch['data']
            labels = batch['label']
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

        # 遍历 DataLoader 再次获取所有样本的特征和标签
        all_features = []  # 存储所有特征
        all_labels = []  # 存储所有标签
        all_imgpaths = []  # 存储所有图像路径

        for batch in tqdm(data_loader, desc="Gathering all samples"):
            images = batch['data'].cuda()  # 移动到 GPU
            labels = batch['label'].cpu().numpy()  # 确保标签在 CPU 上
            imgpaths = batch['imgpath']  # 获取图像路径

            # 确保输入是连续的并转换为浮点型
            images = images.contiguous().float()

            # 获取当前批次的特征
            with torch.no_grad():
                features = _network.extract_vector(images).cpu().numpy()

            # 将当前批次的特征和标签添加到列表中
            all_features.append(features)
            all_labels.append(labels)
            all_imgpaths.append(imgpaths)

        # 最后合并所有特征和标签
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_imgpaths = list(itertools.chain.from_iterable(all_imgpaths))
        # all_imgpaths = np.concatenate(all_imgpaths, axis=0)  # 如果需要合并图像路径
        # print(all_imgpaths)

        # 计算每个类的最近 m 个样本
        for class_id in tqdm(range(start_class, known_class), desc="Selecting nearest samples"):
            class_indices = np.where(all_labels == class_id)[0]
            if len(class_indices) == 0:
                continue

            class_center = class_means[class_id]

            # 计算到类中心的距离
            distances = np.linalg.norm(all_features[class_indices] - class_center, axis=1)

            # 获取距离最小的 m 个样本的索引
            nearest_indices = np.argsort(distances)[:m]
            selected_indices[class_id] = class_indices[nearest_indices].tolist()
            selected_features[class_id] = all_features[class_indices][nearest_indices]

        retained_images = []
        retained_labels = []
        for class_id in range(start_class, known_class):
            indices = selected_indices[class_id]
            retained_images.extend([all_imgpaths[i] for i in indices])
            retained_labels.extend([all_labels[i] for i in indices])

        retained_datasets = BaseDataset(loader=trainer.train_loader.dataset.loader,
                                        transform=trainer.train_loader.dataset.transform)

        retained_datasets.images = retained_images
        retained_datasets.labels = retained_labels

        # 新类
        all_remaining_images = []
        all_remaining_labels = []
        for class_id in range(known_class, total_class):
            class_indices = np.where(all_labels == class_id)[0]
            all_remaining_images.extend([all_imgpaths[i] for i in class_indices])
            all_remaining_labels.extend([all_labels[i] for i in class_indices])
        all_remaining_datasets = BaseDataset(loader=trainer.train_loader.dataset.loader,
                                             transform=trainer.train_loader.dataset.transform)
        all_remaining_datasets.images = all_remaining_images
        all_remaining_datasets.labels = all_remaining_labels

        return retained_datasets, all_remaining_datasets

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))
    def forward(self, x):
        return self.alpha * x + self.beta
    def printParam(self, i):
        print(i, self.alpha.item(), self.beta.item())

@ALGORITHMS.register
class BiC(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.args = trainer.cfg
        self.trainer = trainer

        self._network = None
        self._old_network = None
        self.transform = None
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        self.hook = BiCHook()
        self.bias_layers = nn.ModuleList([BiasLayer()] * (self.args.CIL.sessions + 1))
        self.T = 2
        trainer.register_hook(self.hook)

    def bias_forward(self, x):
        out = []
        b = self.args.CIL.base_classes
        w = self.args.CIL.way
        for s in range(self.args.CIL.sessions):
            if s > 0:
                out.append(self.bias_layers[s](x[:, b + (s - 1) * w:b + s * w]))
            else:
                out.append(self.bias_layers[s](x[:, :b]))
        return torch.cat(out, dim=1)

    def train_step(self, trainer, data, label, attribute, imgpath):
        """
        base train for bic method
        Args:
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """
        session = trainer.session
        self._network = trainer.model
        self._network.train()
        if session > 0:
<<<<<<< HEAD
            self._old_network = trainer.buffer['old_model']
=======
            self._old_network = trainer.buffer["old_model"]
>>>>>>> 266289a9334291351ac9ac88159866e6b72faf8d
            self._old_network = self._old_network.cuda()
            self._old_network.eval()
            loss, acc, per_acc = self.loss_process(data, label, session, distill=True)
        else:
            loss, acc, per_acc = self.loss_process(data, label, session, distill=False)
        ret = {'loss': loss, 'acc': acc, 'per_class_acc': per_acc}
        return ret

    def val_step(self, trainer, data, label, *args, **kwargs):
        """Validation step for bic method.

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
        session = trainer.session
        self._network = trainer.model
        self._network.eval()
        for _ in range(len(self.bias_layers)):
            self.bias_layers[_].train()
        loss, acc, per_acc = self.loss_process(data, label, session, distill=False)
        ret = {'loss': loss.item(), 'acc': acc.item(), 'per_class_acc': per_acc}
        return ret

    def test_step(self, trainer, data, label, *args, **kwargs):
        session = trainer.session
        self._network = trainer.model
        self._network.eval()
        loss, acc, per_acc = self.loss_process(data, label, session, distill=False)
        ret = {'loss': loss.item(), 'acc': acc.item(), 'per_class_acc': per_acc}
        return ret

    def get_net(self):
        return self._network

    def loss_process(self, data, label, session, distill=False):
        known_class = self.args.CIL.base_classes + session * self.args.CIL.way
        total_class = known_class + self.args.CIL.way
        data = data.cuda()
        labels = label.cuda()
        logits = self._network(data)
        logits = self.bias_forward(logits)
        logits_ = logits[:, :known_class]
        acc = accuracy(logits_, labels)[0]
        per_acc = str(per_class_accuracy(logits_, labels))
        clf_loss = self.loss(logits_, labels)
        if distill:
            lamda = known_class / total_class
            with torch.no_grad():
                old_logits = self._old_network(data)
                old_logits = self.bias_forward(old_logits)
                old_logits_ = old_logits[:, :known_class]
                old_logits = F.softmax(old_logits_ / self.T, dim=1)
            log_logits = F.log_softmax(old_logits_ / self.T, dim=1)
            distill_loss = -torch.mean(torch.sum(old_logits * log_logits, dim=1))
            loss = distill_loss * lamda + clf_loss * (1 - lamda)
        else:
            loss = clf_loss
        loss.backward()
        return loss, acc, per_acc
