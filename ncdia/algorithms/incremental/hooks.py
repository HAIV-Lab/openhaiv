import os
from tqdm import tqdm
import torch
from torch import optim
import copy
from torch.utils.data import DataLoader
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
from ncdia.models.net.alice_net import AliceNET

@HOOKS.register
class iCaRLHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()
    


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