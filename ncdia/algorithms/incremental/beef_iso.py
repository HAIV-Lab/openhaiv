import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import itertools

from ncdia.utils import ALGORITHMS
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
from ncdia.algorithms.base import BaseAlg
from ncdia.dataloader import MergedDataset
from ncdia.dataloader import BaseDataset
from torch.utils.data import DataLoader
from ncdia.models.net.alice_net import AliceNET
from ncdia.utils.losses import AngularPenaltySMLoss
from ncdia.utils.metrics import accuracy, per_class_accuracy

EPSILON = 1e-8


@HOOKS.register
class BeefIsoHook(AlgHook):
    def __init__(self, trainer) -> None:
        super().__init__()
        self.trainer = trainer  # Store trainer in the hook
        # self._network = trainer.algorithm._network
        self._fix_memory = True
        self.args = trainer.cfg  # Access the trainer's configuration
        self.alg = None

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
            
    def before_train(self, trainer) -> None:

        session = self.trainer.session  # Access trainer's session
        self.alg = trainer.algorithm
        # if session == 0:
        #     self.net.known_class = 0
        # else:
        #     self.net.known_class = self.args.CIL.base_classes + (session - 1) * self.args.CIL.way
        self.alg._total_classes = self.args.CIL.base_classes + session * self.args.CIL.way
        self.alg._known_classes = self.alg._total_classes - self.args.CIL.way
        # print("way", self.args.CIL.way)
        print("before_train")
        self.trainer.model.task_sizes = [20 ]*session
        print(session)
        print(self.alg._known_classes, self.alg._total_classes)
        print(self.trainer.model.biases)
        
        # print("session: ", session)
        # print("self.known_class: ", self.alg._known_classes)
        # print("self.total_class: ", self.alg._total_classes)
        
        algorithm = trainer.algorithm
        
        # trainer.train_loader
        # _hist_trainset = trainer.hist_trainset
        # now_dataset = trainer.train_loader.dataset
        # _hist_trainset = MergedDataset([_hist_trainset], replace_transform=True)    

        # if self._fix_memory and trainer.session >=1:
        #     _hist_trainset , new_dataset = self.construct_exemplar_unified(_hist_trainset, trainer.cfg.CIL.per_classes, trainer)
        #     _hist_trainset = MergedDataset([_hist_trainset], replace_transform=True)
        #     new_dataset = MergedDataset([new_dataset], replace_transform=True)

        # if trainer.session >=1:
        #     _hist_trainset.merge([new_dataset], replace_transform=True)
        # else:
        #     _hist_trainset.merge([now_dataset], replace_transform=True)
        # # print(_hist_trainset.labels)
        # trainer._train_loader = DataLoader(_hist_trainset, **trainer._train_loader_kwargs)

        # # val_loader
        # trainer.val_loader
        # _hist_valset = MergedDataset([trainer.hist_valset], replace_transform=True)
        # _hist_valset.merge([trainer.val_loader.dataset], replace_transform=True)
        # trainer._val_loader = DataLoader(_hist_valset, **trainer._val_loader_kwargs)
        # # trainer.model.update_fc(trainer.cfg.CIL.way)
        
        # self.trainer.model.task_sizes.append(self.alg._total_classes - self.alg._known_classes)
        if session > 0:
            self.trainer.model.update_fc_before(self.alg._total_classes)
            # self.trainer.model.task_sizes.append(self.alg._total_classes - self.alg._known_classes)
        # print("here")
        # print(self.trainer.model)
        self.trainer._network_module_ptr = self.trainer.model

        if self.trainer.session > 0:
            for id in range(self.trainer.session):
                for p in self.trainer.model.convnets[id].parameters():
                    p.requires_grad = False
            for p in self.trainer.model.old_fc.parameters():
                p.requires_grad = False

    def after_train(self, trainer) -> None:
        trainer.update_hist_dataset(
            key = 'hist_trainset',
            new_dataset =  trainer.train_loader.dataset,
            replace_transform=True,
            inplace=True
        )

        trainer.update_hist_dataset(
            key = 'hist_testset',
            new_dataset = trainer.val_loader.dataset,
            replace_transform=True,
            inplace=True
        )
        # if self.reduce_batch_size:
        #     if self._cur_task == 0:
        #         self.args["batch_size"] = self.args["batch_size"]
        #     else:
        #         self.args["batch_size"] = self.args["batch_size"] * (self._cur_task + 1) // (self._cur_task + 2)
    
    def after_test(self, trainer) -> None:
        print("----------after_test----------")
        self.trainer._network_module_ptr.update_fc_after()
        # self.alg._known_classes = self.alg._total_classes
        # self.alg._total_classes += 20
        # print(self.alg._known_classes, self.alg._total_classes)


@ALGORITHMS.register
class BEEFISO(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.args = trainer.cfg
        self.trainer = trainer

        self._network = None
        self.transform = None
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        self.hook = BeefIsoHook(trainer)  # Pass trainer to the hook
        trainer.register_hook(self.hook)
        session = trainer.session
        self.sinkhorn_reg = self.args.sinkhorn
        self.calibration_term = self.args.calibration_term
        self._network_copy = None
        self._known_classes = 0
        self._total_classes = 0
    
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
        self._network = trainer.model
        self._network.train()

        data = data.cuda()
        labels = label.cuda()
        self._network = self._network.cuda()
        loss, acc, per_acc = self._compute_loss(data, labels)
        loss.backward()

        ret = {'loss': loss, 'acc': acc, 'per_class_acc': per_acc}
        return ret

    def val_step(self, trainer, data, label, *args, **kwargs):
        print("------------val_step----------")
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
        logits = self._network(data)["logits"]
        logits_ = logits[:, :test_class]
        acc = accuracy(logits_, labels)[0]
        loss = self.loss(logits_, labels)
        per_acc = str(per_class_accuracy(logits_, labels))

        ret = {'loss': loss.item(), 'acc': acc.item(), 'per_class_acc': per_acc}
        
        return ret
    
    def test_step(self, trainer, data, label, *args, **kwargs):
        print("----------test_step----------")
        return self.val_step(trainer, data, label, *args, **kwargs)

    
    def target2onehot(self, targets, n_classes):
        onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
        onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
        return onehot
    
    def _compute_loss(self, data, labels):
        session = self.trainer.session
        if session == 0:
            known_class = self.args.CIL.base_classe
        else:
            known_class = self.args.CIL.base_classes + (session - 1) * self.args.CIL.way
        total_class = self.args.CIL.base_classes + session * self.args.CIL.way  # 40
        
        logits = self._network(data)["logits"]
        # print(self._network)
        # print(logits)
        # print(logits.shape)
        # print(self._total_classes)
        logits_ = logits[:, :self._total_classes]
        
        # print(self.args)
        self.energy_weight = 0.01
        loss_en = self.energy_weight * self.get_energy_loss(data,labels,labels)
        loss = F.cross_entropy(logits_, labels)
        loss = loss + loss_en
        
        # if session == 2:
            # print(logits_.shape, labels.shape)
            # print(labels)
            # print("1")
            # print(logits_)
            # print("2")
        acc = accuracy(logits_, labels)[0]
        per_acc = str(per_class_accuracy(logits_, labels))

        return loss, acc, per_acc
    
    def get_energy_loss(self,inputs,targets,pseudo_targets):
        inputs = self.sample_q(inputs)
        
        out = self._network(inputs)
        if self.trainer.session == 0:
            targets = targets + self._total_classes
            train_logits, energy_logits = out["logits"], out["energy_logits"]
        else:
            targets = targets + (self._total_classes - self._known_classes) + self.trainer.session
            train_logits, energy_logits = out["train_logits"], out["energy_logits"]
        
        logits = torch.cat([train_logits,energy_logits],dim=1)
        
        logits[:,pseudo_targets] = 1e-9        
        energy_loss = F.cross_entropy(logits,targets)
        return energy_loss
    
    def sample_q(self, replay_buffer, n_steps=3):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        self._network_copy = self.trainer._network_module_ptr.copy().freeze()
        init_sample = replay_buffer
        init_sample = torch.rot90(init_sample, 2, (2, 3))
        embedding_k = init_sample.clone().detach().requires_grad_(True)
        optimizer_gen = torch.optim.SGD(
            [embedding_k], lr=1e-2)
        for k in range(1, n_steps + 1):
            out = self._network_copy(embedding_k)
            if self.trainer.session == 0:
                energy_logits, train_logits = out["energy_logits"], out["logits"]
            else:
                energy_logits, train_logits = out["energy_logits"], out["train_logits"]
            num_forwards = energy_logits.shape[1]
            logits = torch.cat([train_logits,energy_logits],dim=1)
            negative_energy = torch.log(torch.sum(torch.softmax(logits,dim=1)[:,-num_forwards:]))
            optimizer_gen.zero_grad()
            negative_energy.sum().backward()
            optimizer_gen.step()
            embedding_k.data += 1e-3 * \
                torch.randn_like(embedding_k)
        final_samples = embedding_k.detach()
        return final_samples