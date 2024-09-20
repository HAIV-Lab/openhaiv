from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch


@HOOKS.register
class FACTHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        algorithm.replace_fc()


@HOOKS.register
class AliceHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        algorithm.replace_fc()


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


@HOOKS.register
class SAVCHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
    
    def after_train(self, trainer) -> None:
        algorithm = trainer.algorithm
        algorithm.replace_fc()
    
    def before_train(self, trainer) -> None:
        trainer.train_loader.dataset.multi_train = True
