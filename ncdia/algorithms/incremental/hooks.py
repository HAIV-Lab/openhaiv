import os
import logging
import numpy as np 
from tqdm import tqdm
import itertools
import torch
from torch import optim
import copy
from torch.utils.data import DataLoader
from ncdia.dataloader import MergedDataset
from ncdia.dataloader import BaseDataset
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
from ncdia.models.net.alice_net import AliceNET


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
        trainer.update_hist_testset(
            trainer.test_loader.dataset,
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
                features = _network.get_features(images).cpu().numpy()

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



@HOOKS.register
class iCaRLHook(AlgHook):
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
        all_features = []  # 存储所有特征
        all_labels = []    # 存储所有标签
        all_imgpaths = []  # 存储所有图像路径

        for batch in tqdm(data_loader, desc="Gathering all samples"):
            images = batch['data'].cuda()  # 移动到 GPU
            labels = batch['label'].cpu().numpy()  # 确保标签在 CPU 上
            imgpaths = batch['imgpath']  # 获取图像路径

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


    def before_test(self, trainer) -> None:
        """
        在进入for epoch in range(max_epochs)循环之前，对测试数据集进行处理。
        """
        trainer.test_loader
        _hist_testset = MergedDataset([trainer.hist_testset], replace_transform=True)
        _hist_testset.merge([trainer.test_loader.dataset], replace_transform=True)
        trainer._test_loader = DataLoader(_hist_testset, **trainer._test_loader_kwargs)

    def after_test(self, trainer) -> None:
        """
        在测试结束后，将当前session中需要保存的数据保存到hist_testset中。
        """
        trainer.update_hist_testset(
            trainer.test_loader.dataset,
            replace_transform=True,
            inplace=True
        )
    


@HOOKS.register
class IL2AHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()


@HOOKS.register
class FinetuneHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        filename = 'task_' + str(trainer.session) + '.pth'
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))



@HOOKS.register
class WAHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
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
        session = trainer.session
        if session > 0:
            self.weight_align(trainer)
        
    
    def weight_align(self, trainer):
        session = trainer.session
        known_classes = trainer.cfg.CIL.base_classes + (session-1) * trainer.cfg.CIL.way
        increment = trainer.cfg.CIL.num_classes - known_classes

        weights = trainer.model.fc.weight

        newnorm = torch.norm(weights[:, -increment:], p=2, dim=1)
        oldnorm = torch.norm(weights[:, :-increment], p=2, dim=1)

        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        trainer.model.fc.weight.data[:, -increment:] *= gamma


@HOOKS.register
class EWCHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        filename = 'task_' + str(trainer.session) + '.pth'
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))
        self.update_fisher(trainer)
    
    def update_fisher(self, trainer):
        session = trainer.session
        if session == 0:
            fisher = self.getFisherDiagonal(trainer.train_loader, trainer)
            filename = 'fisher task_ ' + str(trainer.session) + '.pth'
            torch.save(fisher, os.path.join(trainer.work_dir, filename))
        else:
            known_classes = trainer.cfg.CIL.base_classes + session * trainer.cfg.CIL.way
            alpha = known_classes / trainer.cfg.CIL.num_classes
            fisher = self.fisher = torch.load(os.path.join(trainer.work_dir, 'fisher task_ ' + str(trainer.session - 1) + '.pth'))
            new_fisher = self.getFisherDiagonal(trainer.train_loader, trainer)
            for n, p in new_fisher.items():
                new_fisher[n][: len(fisher[n])] = (
                    alpha * fisher[n]
                    + (1 - alpha) * new_fisher[n][: len(fisher[n])]
                )
            fisher = new_fisher
            filename = 'fisher task_ ' + str(trainer.session) + '.pth'
            torch.save(fisher, os.path.join(trainer.work_dir, filename))
        mean = {
            n: p.clone().detach()
            for n, p in trainer.model.named_parameters()
            if p.requires_grad
        }
        mean_name = 'mean task_ ' + str(trainer.session) + '.pth'
        torch.save(mean, os.path.join(trainer.work_dir, mean_name))
 

        
    def getFisherDiagonal(self, train_loader, trainer):
        fisher = {
            n: torch.zeros(p.shape).cuda()
            for n, p in  trainer.model.named_parameters()
            if p.requires_grad
        }

        trainer.model.train()
        optimizer = optim.SGD(trainer.model.parameters(), lr=0.1)
        for i, batch in enumerate(train_loader):
            inputs = batch['data'].cuda()
            targets = batch['label'].cuda()
            logits = trainer.model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            for n, p in trainer.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(0.0001))

        return fisher

@HOOKS.register
class LwFHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
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
            


@HOOKS.register
class FACTHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        algorithm.replace_fc()
        

        filename = 'task_' + str(trainer.session) + '.pth'
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))
        if trainer.session == 0:
            self.save_train_static(trainer)
    
    def save_train_static(self, trainer):
        all_class = trainer.train_loader.dataset.num_classes
        features, logits, labels = [], [], []
        tbar = tqdm(trainer.train_loader, dynamic_ncols=True, disable=True)
        for batch in tbar:
            data = batch['data'].to(trainer.device)
            label = batch['label'].to(trainer.device)
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
        filename = 'train_static.pt'
        torch.save({'train_features': features, 'train_logits': logits, 'prototype': torch.stack(prototype_cls)}, os.path.join(trainer.work_dir, filename))


@HOOKS.register
class AliceHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        algorithm.replace_fc()

        filename = 'task_' + str(trainer.session) + '.pth'
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))
        if trainer.session == 0:
            self.save_train_static(trainer)
    
    def save_train_static(self, trainer):
        all_class = trainer.train_loader.dataset.num_classes
        features, logits, labels = [], [], []
        tbar = tqdm(trainer.train_loader, dynamic_ncols=True, disable=True)
        for batch in tbar:
            data = batch['data'].to(trainer.device)
            label = batch['label'].to(trainer.device)
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
        filename = 'train_static.pt'
        torch.save({'train_features': features, 'train_logits': logits, 'prototype': torch.stack(prototype_cls)}, os.path.join(trainer.work_dir, filename))

@HOOKS.register
class SAVCHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        algorithm.replace_fc()

        filename = 'task_' + str(trainer.session) + '.pth'
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))
        if trainer.session == 0:
            self.save_train_static(trainer)
    
    def save_train_static(self, trainer):
        all_class = trainer.train_loader.dataset.num_classes
        features, logits, labels, att_logits = [], [], [], []
        tbar = tqdm(trainer.train_loader, dynamic_ncols=True, disable=True)
        for batch in tbar:
            data = batch['data'].to(trainer.device)
            label = batch['label'].to(trainer.device)
            joint_preds, joint_preds_att = trainer.model(data)
            joint_preds = joint_preds[:, :all_class]
            feats = trainer.model.get_features(data)

            att_logits.append(joint_preds_att.clone().detach().cpu())
            features.append(feats.clone().detach().cpu())
            logits.append(joint_preds.clone().detach().cpu())
            labels.append(label.clone().detach().cpu())

        features = torch.cat(features, dim=0)
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0).to(torch.int)
        att_logits = torch.cat(att_logits, dim=0)

        classes = torch.unique(labels)
        prototype_att = []
        prototype_cls = []
        for cls in classes:
            cls_indices = torch.where(labels == cls)
            cls_preds = logits[cls_indices]
            att_predictions = att_logits[cls_indices]
            prototype_cls.append(torch.mean(cls_preds, dim=0))
            prototype_att.append(torch.mean(att_predictions, dim=0))

        att_logits = self.get_test_logits(trainer)
        filename = 'train_static.pt'
        torch.save({'train_features': features, 'train_logits': logits, 'att_logits': att_logits, 'prototype': torch.stack(prototype_cls), 'prototype_att': torch.stack(prototype_att)}, os.path.join(trainer.work_dir, filename))
    
    def before_train(self, trainer) -> None:
        trainer.train_loader.dataset.multi_train = True\
    
    def get_test_logits(self, trainer) -> None:
        all_class = trainer.train_loader.dataset.num_classes
        att_logits = []
        tbar = tqdm(trainer.test_loader, dynamic_ncols=True, disable=True)
        for batch in tbar:
            data = batch['data'].to(trainer.device)
            label = batch['label'].to(trainer.device)
            joint_preds, joint_preds_att = trainer.model(data)
            joint_preds = joint_preds[:, :all_class]

            att_logits.append(joint_preds_att.clone().detach().cpu())
        
        att_logits = torch.cat(att_logits, dim=0)
        return att_logits

@HOOKS.register
class AliceHook_s(AlgHook):
    def __init__(self) -> None:
        super().__init__()

    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        algorithm.replace_fc()
        filename = 'task_' + str(trainer.session) + '.pth'
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))
        if trainer.session == 0:
            self.save_train_static(trainer)

    def save_train_static(self, trainer):
        all_class = trainer.train_loader.dataset.num_classes
        features, logits, labels = [], [], []
        tbar = tqdm(trainer.train_loader, dynamic_ncols=True, disable=True)
        for batch in tbar:
            data = batch['data'].to(trainer.device)
            label = batch['label'].to(trainer.device)
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
        filename = 'train_static.pt'
        torch.save({'train_features': features, 'train_logits': logits, 'prototype': torch.stack(prototype_cls)}, os.path.join(trainer.work_dir, filename))