import os
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn

from ncdia.utils import ALGORITHMS
from ncdia.algorithms.base import BaseAlg
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
from ncdia.dataloader import MergedDataset, BaseDataset
from ncdia.utils.metrics import accuracy, per_class_accuracy
import logging
import numpy as np
from sklearn.svm import LinearSVC

import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def filter_dataloader_by_class(train_loader, test_loader, target_classes):
    """
    从训练DataLoader中筛选出特定类别的样本，并创建一个应用测试转换的新DataLoader。

    Args:
        train_loader (DataLoader): 原始训练DataLoader。
        test_loader (DataLoader): 提供目标转换的测试DataLoader。
        target_classes (list): 需要筛选的类别列表。

    Returns:
        DataLoader: 包含筛选后样本并应用测试转换的新DataLoader。
    """
    train_dataset = train_loader.dataset
    if hasattr(train_dataset, 'targets'):
        all_labels = train_dataset.targets
    elif hasattr(train_dataset, 'labels'):
        all_labels = train_dataset.labels
    else:
        all_labels = []
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            if isinstance(sample, dict):
                all_labels.append(sample['label'])
            else:
                _, label = sample
                all_labels.append(label)
        all_labels = torch.tensor(all_labels)
    all_labels = torch.as_tensor(all_labels)
    target_classes = torch.as_tensor(target_classes)
    mask = torch.isin(all_labels, target_classes)
    indices = torch.nonzero(mask, as_tuple=True)[0]
    filtered_subset = Subset(train_dataset, indices)
    def pil_loader(path):
        """
        Ref:
        https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
        """
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            dd = self.subset[idx]
            # data = dd['data']
            label = dd['label']
            imgpath = dd['imgpath']
            if self.transform:
                data = self.transform(pil_loader(imgpath))
            return data, label
    test_transform = test_loader.dataset.transform
    transformed_dataset = TransformedSubset(filtered_subset, test_transform)
    filtered_dataloader = DataLoader(
        transformed_dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        drop_last=False
    )
    
    return filtered_dataloader

@HOOKS.register
class FeTrILHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()

    def before_train(self, trainer) -> None:
        """
        在进入for epoch in range(max_epochs)循环之前，对训练数据集进行处理。
        """
        session = trainer.session
        if session > 0:
            trainer.algorithm._compute_means(trainer)
            trainer.algorithm._compute_relations()
            trainer.algorithm._build_feature_set(trainer.train_loader, trainer.test_loader)
            trainer._train_loader = DataLoader(trainer.algorithm._feature_trainset, batch_size=trainer.train_loader.batch_size, shuffle=True, num_workers=trainer.train_loader.num_workers) 
    
    def before_epoch(self, trainer) -> None:
        """
        在进入for batch_idx, (data, label) in enumerate(train_loader)循环之前，对训练数据集进行处理。
        """
        trainer.algorithm._means = trainer._means
        trainer.algorithm._svm_accs = trainer._svm_accs

    def after_train(self, trainer) -> None:
        """
        在训练结束后，将当前session中需要保存的数据保存到hist_trainset中。
        """
        session = trainer.session
        if session == 0:
            trainer.algorithm._compute_means(trainer)
            trainer.algorithm._build_feature_set(trainer.train_loader, trainer.test_loader)
        
        print("train svm")
        trainer.algorithm._train_svm(trainer)


@ALGORITHMS.register
class FeTrIL(BaseAlg):
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.trainer = trainer

        hook = FeTrILHook()
        trainer.register_hook(hook)
        self._network = None
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        session = trainer.session
        known_class =  max(self.args.CIL.base_classes + (session - 1) * self.args.CIL.way, 0)
        self._known_classes = known_class
        self._total_classes = self.args.CIL.base_classes + (session) * self.args.CIL.way

        self._means = None
        self._relations = []
        self._svm_accs = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._means = [] if session == 0 else np.load(os.path.join('temp/', 'means.npy'), allow_pickle=True).tolist()
        self._svm_accs = [] if session == 0 else np.load(os.path.join('temp/','svm_accs.npy'), allow_pickle=True).tolist()

        self._network = trainer.model
    
    def train_step(self, trainer, data, label, attribute, imgpath):
        """
        base train for fact method
        Args:
            data: data in batch
            label: label in batch
            attribute: attribute in batch
            imgpath: imgpath in batch
        """
        if hasattr(self._network,"module"):
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network

        session = trainer.session

        known_class =  max(self.args.CIL.base_classes + (session - 1) * self.args.CIL.way, 0)
        self._known_classes = known_class
        self._total_classes = self.args.CIL.base_classes + (session) * self.args.CIL.way

        self._network = trainer.model
        self._network.train()

        data = data.cuda()
        labels = label.cuda()
        data = data.to(self._network_module_ptr.fc.weight.dtype)
        if session ==0:
            logits = self._network(data)
        else:
            logits = self._network_module_ptr.fc(data)
        logits_ = logits[:, :self._total_classes]
        acc = accuracy(logits_, labels)[0]
        per_acc = str(per_class_accuracy(logits_, labels))
        loss = self.loss(logits_, labels)
        loss.backward()

        ret = {}
        ret['loss'] = loss
        ret['acc'] = acc
        ret['per_class_acc'] = per_acc

        return ret

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.get_features(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.get_features(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
    
    def _compute_means(self, trainer):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                # data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                #                                                     mode='test', ret_data=True)
                # idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                idx_loader = filter_dataloader_by_class(trainer.train_loader, trainer.test_loader, [class_idx])
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._means.append(class_mean)
            
            if not os.path.exists('temp/'):
                os.makedirs('temp/')
            np.save('temp/means.npy', self._means)
            print("build _means done")

    def _compute_relations(self):
        old_means = np.array(self._means[:self._known_classes])
        new_means = np.array(self._means[self._known_classes:])
        self._relations=np.argmax((old_means/np.linalg.norm(old_means,axis=1)[:,None])@(new_means/np.linalg.norm(new_means,axis=1)[:,None]).T,axis=1)+self._known_classes


    def _build_feature_set(self, train_loader, test_loader):
        self.vectors_train = []
        self.labels_train = []
        for class_idx in range(self._known_classes, self._total_classes):
            target_classes = [class_idx]
            filtered_loader = filter_dataloader_by_class(self.trainer.train_loader, self.trainer.test_loader, target_classes)
            vectors, _ = self._extract_vectors(filtered_loader)
            self.vectors_train.append(vectors)
            self.labels_train.append([class_idx]*len(vectors))
        for class_idx in range(0,self._known_classes):
            new_idx = self._relations[class_idx]
            self.vectors_train.append(self.vectors_train[new_idx-self._known_classes]-self._means[new_idx]+self._means[class_idx])
            self.labels_train.append([class_idx]*len(self.vectors_train[-1]))
        
        self.vectors_train = np.concatenate(self.vectors_train)
        self.labels_train = np.concatenate(self.labels_train)
        self._feature_trainset = FeatureDataset(self.vectors_train,self.labels_train)
        
        self.vectors_test = []
        self.labels_test = []
        for class_idx in range(0, self._total_classes):
            target_classes = [class_idx]
            filtered_loader = filter_dataloader_by_class(self.trainer.test_loader, self.trainer.test_loader, target_classes)
            vectors, _ = self._extract_vectors(filtered_loader)
            self.vectors_test.append(vectors)
            self.labels_test.append([class_idx]*len(vectors))
        self.vectors_test = np.concatenate(self.vectors_test)
        self.labels_test = np.concatenate(self.labels_test)

        self._feature_testset = FeatureDataset(self.vectors_test,self.labels_test)

    
    def _train_svm(self, trainer):
        train_set, test_set= self._feature_trainset, self._feature_testset
        train_features = train_set.features.numpy()
        train_labels = train_set.labels.numpy()
        test_features = test_set.features.numpy()
        test_labels = test_set.labels.numpy()
        train_features = train_features/np.linalg.norm(train_features,axis=1)[:,None]
        test_features = test_features/np.linalg.norm(test_features,axis=1)[:,None]
        svm_classifier = LinearSVC(random_state=42)
        svm_classifier.fit(train_features,train_labels)
        logging.info("svm train: acc: {}".format(np.around(svm_classifier.score(train_features,train_labels)*100,decimals=2)))
        acc = svm_classifier.score(test_features,test_labels)
        self._svm_accs.append(np.around(acc*100,decimals=2))
        logging.info("svm evaluation: acc_list: {}".format(self._svm_accs))
        
        if not os.path.exists('temp/'):
            os.makedirs('temp/')
        np.save('temp/svm_accs.npy', self._svm_accs)
        print("build svm_accs done")

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
    
    def get_net(self):
            return self._network
    

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels), "Data size error!"
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return idx, feature, label