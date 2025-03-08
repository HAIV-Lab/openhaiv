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
T = 2

@HOOKS.register
class COILHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
        self._fix_memory = True
    def before_train(self, trainer) -> None:
        trainer.train_loader
        _hist_trainset = trainer.hist_trainset
        now_dataset = trainer.train_loader.dataset
        _hist_trainset = MergedDataset([_hist_trainset], replace_transform=True)    

        if self._fix_memory and trainer.session >=1:
            _hist_trainset , new_dataset = self.construct_exemplar_unified(_hist_trainset, trainer.cfg.CIL.per_classes, trainer)
            _hist_trainset = MergedDataset([_hist_trainset], replace_transform=True)
            new_dataset = MergedDataset([new_dataset], replace_transform=True)

        if trainer.session >=1:
            _hist_trainset.merge([new_dataset], replace_transform=True)
        else:
            _hist_trainset.merge([now_dataset], replace_transform=True)
        # print(_hist_trainset.labels)
        trainer._train_loader = DataLoader(_hist_trainset, **trainer._train_loader_kwargs)

        # val_loader
        trainer.val_loader
        _hist_valset = MergedDataset([trainer.hist_valset], replace_transform=True)
        _hist_valset.merge([trainer.val_loader.dataset], replace_transform=True)
        trainer._val_loader = DataLoader(_hist_valset, **trainer._val_loader_kwargs)
    
    def after_train(self, trainer) -> None:
        trainer.update_hist_trainset(
            trainer.train_loader.dataset,
            replace_transform=True,
            inplace=True
        )

        trainer.update_hist_valset(
            trainer.val_loader.dataset,
            replace_transform=True,
            inplace=True
        )
        algorithm = trainer.algorithm
        filename = 'task_' + str(trainer.session) + '.pth'
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))
        trainer.old_model = AliceNET(
            trainer.cfg.model.network,
            trainer.cfg.CIL.base_classes,
            trainer.cfg.CIL.num_classes,
            trainer.cfg.CIL.att_classes,
            trainer.cfg.model.net_alice
        )
        trainer.old_model.load_state_dict(trainer.model.state_dict())
        for param in trainer.old_model.parameters():
            param.requires_grad = False
        _ot_new_branch = self.solving_ot(trainer)
        branch_path = 'task_' + str(trainer.session) + 'branch' + '.pt'
        torch.save(_ot_new_branch, os.path.join(trainer.work_dir, branch_path))

        trainer.branch = _ot_new_branch

    def construct_exemplar_unified(self, trainset, m, trainer):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        
        trainset.merge([trainer.train_loader.dataset], replace_transform=True)
        args = trainer.cfg
        session = trainer.session
        total_class = args.CIL.base_classes + (session) * args.CIL.way
        known_class =  max(args.CIL.base_classes + (session - 1) * args.CIL.way, 0)
        start_class = max(args.CIL.base_classes + (session - 2) * args.CIL.way, 0)
        _network = trainer.model
        _feature_dim = _network.num_features
        class_means = np.zeros((total_class, _feature_dim))

        
        data_loader = DataLoader(trainset, **trainer._train_loader_kwargs)
        class_sums = np.zeros((total_class, _feature_dim))
        class_counts = np.zeros(total_class)
        
        for i, batch in enumerate(tqdm(data_loader, desc="Calculating class means")):
            images = batch['data']
            labels = batch['label']
            images = images.cuda()
            with torch.no_grad():
                features = _network.get_features(images)
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
        all_features = []  
        all_labels = []    
        all_imgpaths = []  

        for batch in tqdm(data_loader, desc="Gathering all samples"):
            images = batch['data'].cuda()  
            labels = batch['label'].cpu().numpy()  
            imgpaths = batch['imgpath']  

            # 确保输入是连续的并转换为浮点型
            images = images.contiguous().float()

            # 获取当前批次的特征
            with torch.no_grad():
                features = _network.get_features(images).cpu().numpy()

            # 将当前批次的特征和标签添加到列表中
            all_features.append(features)
            all_labels.append(labels)
            all_imgpaths.append(imgpaths)

        # 最后合并所有特征和标签
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_imgpaths = list(itertools.chain.from_iterable(all_imgpaths))
        # all_imgpaths = np.concatenate(all_imgpaths, axis=0)  
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
    
        retained_datasets = BaseDataset(loader = trainer.train_loader.dataset.loader, 
                                        transform = trainer.train_loader.dataset.transform)

        retained_datasets.images = retained_images
        retained_datasets.labels = retained_labels

        # 新类
        all_remaining_images = []
        all_remaining_labels = []
        for class_id in range(known_class, total_class):
            class_indices = np.where(all_labels == class_id)[0]
            all_remaining_images.extend([all_imgpaths[i] for i in class_indices])
            all_remaining_labels.extend([all_labels[i] for i in class_indices])
        all_remaining_datasets = BaseDataset(loader = trainer.train_loader.dataset.loader, 
                                        transform = trainer.train_loader.dataset.transform)
        all_remaining_datasets.images = all_remaining_images
        all_remaining_datasets.labels = all_remaining_labels

        return retained_datasets, all_remaining_datasets
    
    def solving_ot(self, trainer):
        import ot

        session = trainer.session
        self.args = trainer.cfg

        self.sinkhorn_reg = self.args.sinkhorn
        known_class = max(self.args.CIL.base_classes + (session - 1) * self.args.CIL.way, 0)
        total_class = self.args.CIL.base_classes + self.args.CIL.way * session
        with torch.no_grad():
            if total_class == trainer.cfg.CIL.num_classes:
                print("training over, no more ot solving")
                return None
            
            each_time_class_num = self.args.CIL.way
            self._extract_class_means(
                trainer, 0, total_class + each_time_class_num
            )
            
            former_class_means = torch.tensor(
                self._ot_prototype_means[: total_class]
            )
            next_period_class_means = torch.tensor(
                self._ot_prototype_means[
                    total_class : total_class + each_time_class_num
                ]
            )
            Q_cost_matrix = torch.cdist(
                former_class_means, next_period_class_means, p=self.args.norm_term
            )
            # solving ot
            _mu1_vec = (
                torch.ones(len(former_class_means)) / len(former_class_means) * 1.0
            )
            _mu2_vec = (
                torch.ones(len(next_period_class_means)) / len(former_class_means) * 1.0
            )
            # print("mu1_vec :", _mu1_vec)
            # print("mu2_vec :", _mu2_vec)
            # print("q_cost_matrix :", Q_cost_matrix)
            T = ot.sinkhorn(_mu1_vec, _mu2_vec, Q_cost_matrix, self.sinkhorn_reg)
            
            T = torch.tensor(T).float().cuda()
            self._network = trainer.model
            transformed_hat_W = torch.mm(
                T.T, F.normalize(self._network.fc.weight[:total_class], p=2, dim=1)
            )
            oldnorm = torch.norm(self._network.fc.weight[:total_class], p=2, dim=1)
            newnorm = torch.norm(
                transformed_hat_W * len(former_class_means), p=2, dim=1
            )
            meannew = torch.mean(newnorm)
            meanold = torch.mean(oldnorm)
            gamma = meanold / meannew
            self.calibration_term = gamma
            self._ot_new_branch = (
                transformed_hat_W * len(former_class_means) * self.calibration_term
            )
        return transformed_hat_W * len(former_class_means) * self.calibration_term

    def _extract_class_means(self, trainer, low, high):
        _network = trainer.model
        self._ot_prototype_means = np.zeros(
            (trainer.cfg.CIL.num_classes, _network.num_features)
        )
        data_loader = trainer.train_loader
        session = trainer.session
        _feature_dim = _network.num_features
        known_class = max(self.args.CIL.base_classes + (session - 1) * self.args.CIL.way, 0)
        total_class = self.args.CIL.base_classes + self.args.CIL.way * session
        class_means = np.zeros((high, _feature_dim))
        class_sums = np.zeros((high, _feature_dim))
        class_counts = np.zeros(high)

        for i, batch in enumerate(tqdm(data_loader, desc="Calculating class means")):
            images = batch['data']
            labels = batch['label']
            images = images.cuda()
            with torch.no_grad():
                features = _network.get_features(images)
            features_np = features.cpu().numpy()

            for i in range(len(labels)):
                class_index = labels[i].item()
                class_sums[class_index] += features_np[i]
                class_counts[class_index] += 1
        
        for i in range(low, high):
            if class_counts[i] > 0:
                mean = class_sums[i] / class_counts[i]
                class_means[i] = mean
        
        self._ot_prototype_means[low:high,:] = class_means[low:high, :]

@ALGORITHMS.register
class COIL(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.args = trainer.cfg
        self.trainer = trainer

        self._network = None
        self._old_network = None
        self.transform = None
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        self.hook = COILHook()
        trainer.register_hook(self.hook)
        session = trainer.session
        self.sinkhorn_reg = self.args.sinkhorn
        self.calibration_term = self.args.calibration_term
    
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
        known_class = max(self.args.CIL.base_classes + (session - 1) * self.args.CIL.way, 0)
        total_class = self.args.CIL.base_classes + self.args.CIL.way * session
        self.lamda = known_class / total_class

        self._network = trainer.model
        if session >=1:
            self._old_network = trainer.old_model
            self._old_network = self._old_network.cuda()
            self._old_network.eval()
        epoch = trainer.epoch
        epochs = trainer.max_epochs
        weight_ot_init = max(1.0 - (epoch / 2) ** 2, 0)
        weight_ot_co_tuning = (epoch / epochs) ** 2.0

        self._network.train()
        data = data.cuda()
        labels = label.cuda()
        logits = self._network(data)

        onehots = self.target2onehot(labels, total_class)
        logits_ = logits[:, :total_class]
        acc = accuracy(logits_, labels)[0]
        per_acc = str(per_class_accuracy(logits_, labels))
        clf_loss = self.loss(logits_, labels)

        if session >=1:
            branch_path = 'task_' + str(trainer.session-1) + 'branch' + '.pt'
            self._ot_new_branch = torch.load(os.path.join(trainer.work_dir, branch_path))
            self._ot_new_branch = self._ot_new_branch.cuda()
            with torch.no_grad():
                old_logits = self._old_network(data)[:, :known_class]
            hat_pai_k = F.softmax(old_logits / T, dim=1)
            log_pai_k = F.log_softmax(
                logits[:, : known_class] / T, dim=1
            )
            distill_loss = -torch.mean(torch.sum(hat_pai_k * log_pai_k, dim=1))

            epoch = trainer.epoch
            features = F.normalize(self._network.get_features(data), p=2, dim=1)
            
            current_logit_new = F.log_softmax(
                logits[:, known_class:total_class] / T, dim=1
            )
            new_logit_by_wnew_init_by_ot = F.linear(
                features, F.normalize(self._ot_new_branch, p=2, dim=1)
            )
            new_logit_by_wnew_init_by_ot = F.softmax(
                new_logit_by_wnew_init_by_ot / T, dim=1
            )
            new_branch_distill_loss = -torch.mean(
                torch.sum(
                    current_logit_new * new_logit_by_wnew_init_by_ot, dim=1
                )
            )

            loss = (
                distill_loss * self.lamda
                + clf_loss * (1 - self.lamda)
                + 0.001 * (weight_ot_init * new_branch_distill_loss)
            )

        else:
            loss = clf_loss
        loss.backward()
        ret = {}
        ret['loss'] = loss
        ret['acc'] = acc
        ret['per_class_acc'] = per_acc
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
        test_class = self.args.CIL.base_classes + session  * self.args.CIL.way
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
        ret['loss'] = loss.item()
        ret['acc'] = acc.item()
        ret['per_class_acc'] = per_acc
        
        return ret
    
    def test_step(self, trainer, data, label, *args, **kwargs):
        return self.val_step(trainer, data, label, *args, **kwargs)

    
    
    def solving_ot_to_old(self, trainer):
        import ot
        
        current_class_num = self.args.CIL.way
        known_class = max(self.args.CIL.base_classes + (session - 1) * self.args.CIL.way, 0)
        total_class = self.args.CIL.base_classes + self.args.CIL.way * session
        _network = trainer.model
        self._extract_class_means_with_memory(
            trainer, known_class, total_class
        )
        former_class_means = torch.tensor(
            self._ot_prototype_means[: known_class]
        )
        next_period_class_means = torch.tensor(
            self._ot_prototype_means[known_class : total_class]
        )
        Q_cost_matrix = (
            torch.cdist(
                next_period_class_means, former_class_means, p=self.args["norm_term"]
            )
            + EPSILON
        )  # in case of numerical err
        _mu1_vec = torch.ones(len(former_class_means)) / len(former_class_means) * 1.0
        _mu2_vec = (
            torch.ones(len(next_period_class_means)) / len(former_class_means) * 1.0
        )
        T = ot.sinkhorn(_mu2_vec, _mu1_vec, Q_cost_matrix, self.sinkhorn_reg)
        T = torch.tensor(T).float().cuda()
        transformed_hat_W = torch.mm(
            T.T,
            F.normalize(network.fc.weight[-current_class_num:, :], p=2, dim=1),
        )
        return transformed_hat_W * len(former_class_means) * self.calibration_term

    def _extract_class_means_with_memory(self, trainer, low, high):
        pass
    
    def target2onehot(self, targets, n_classes):
        onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
        onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
        return onehot