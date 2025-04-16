import torch
import torch.nn as nn

from ncdia.utils import ALGORITHMS
from ncdia.utils.metrics import accuracy
from ncdia.algorithms.base import BaseAlg
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
import numpy as np
from numpy.linalg import norm
from sklearn.covariance import EmpiricalCovariance
from scipy.special import logsumexp
from tqdm import tqdm
from .metrics import ood_metrics, search_threshold

@ALGORITHMS.register
class MDS(BaseAlg):
    
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.num_classes = trainer.model.num_classes

    def val_step(self, trainer, data, label, *args, **kwargs):
    
        model = trainer.model
        criterion = trainer.criterion
        device = trainer.device

        data, label = data.to(device), label.to(device)
        outputs = model(data)

        loss = criterion(outputs, label)
        acc = accuracy(outputs, label)[0]

        return {"loss": loss.item(), "acc": acc.item()}


    def test_step(self, trainer, data, label, *args, **kwargs):
        
        return self.val_step(trainer, data, label, *args, **kwargs)

    @staticmethod
    def eval(
        id_gt: torch.Tensor,
        id_logits: torch.Tensor, 
        id_feat: torch.Tensor,
        ood_logits: torch.Tensor, 
        ood_feat: torch.Tensor, 
        id_local_logits: torch.Tensor=None, 
        id_local_feat: torch.Tensor=None, 
        ood_local_logits: torch.Tensor=None, 
        ood_local_feat: torch.Tensor=None,
        train_gt: torch.Tensor = None, 
        train_logits: torch.Tensor = None, 
        train_feat: torch.Tensor = None, 
        train_local_logits: torch.Tensor = None, 
        train_local_feat: torch.Tensor = None,
        prototypes: torch.Tensor = None, 
        tpr_th: float = 0.95, 
        prec_th: float = None, 
        hyperparameters = None
    ):
        """
        Args:
            id_gt (torch.Tensor): ID ground truth labels. Shape (N,).
            id_logits (torch.Tensor): ID logits. Shape (N, C).
            id_feat (torch.Tensor): ID features. Shape (N, D).
            ood_logits (torch.Tensor): OOD logits. Shape (M, C).
            ood_feat (torch.Tensor): OOD features. Shape (M, D).
            id_local_logits (torch.Tensor): ID local logits. Shape (N, P, C).
            id_local_feat (torch.Tensor): ID local features. Shape (N, P, D).
            ood_local_logits (torch.Tensor): OOD local logits. Shape (M, P, C).
            ood_local_feat (torch.Tensor): OOD local features. Shape (M, P, D).
            train_gt (torch.Tensor): Training ground truth labels. Shape (K,).
            train_logits (torch.Tensor): Training logits. Shape (K, C).
            train_feat (torch.Tensor): Training features. Shape (K, D).
            train_local_logits (torch.Tensor): Training local logits. Shape (K, P, C).
            train_local_feat (torch.Tensor): Training local features. Shape (K, P, D).
            prototypes (torch.Tensor): Prototypes of train set. Shape (C, C).
            tpr_th (float): True positive rate threshold to compute
                false positive rate. Default is 0.95.
            prec_th (float | None): Precision threshold for searching threshold.
                If None, not searching for threshold. Default is None.
            hyperparameters (dict): Hyperparameters for DPM.
        Returns:
            fpr (float): False positive rate.
            auroc (float): Area under the ROC curve.
            aupr_in (float): Area under the precision-recall curve for in-distribution samples.
            aupr_out (float): Area under the precision-recall curve for out-of-distribution
        """
        class_mean = []
        centered_data = []
        num_classes = max(train_gt.max().item(), ood_gt.max().item(), id_gt.max().item()) + 1
        for c in range(num_classes):
            class_samples = train_feat[train_gt.eq(c)].data
            class_mean.append(class_samples.mean(0))
            centered_data.append(class_samples -
                                    class_mean[c].view(1, -1))

        class_mean = torch.stack(class_mean)

        group_lasso = EmpiricalCovariance(
            assume_centered=False)
        group_lasso.fit(
            torch.cat(centered_data).cpu().numpy().astype(np.float32))
        # inverse of covariance
        precision = torch.from_numpy(group_lasso.precision_).float()

        # logits, features = net(data, return_feature=True)
        pred = id_logits.argmax(1)

        class_scores = torch.zeros((id_logits.shape[0], self.num_classes))
        for c in range(self.num_classes):
            tensor = id_feat.cpu() - self.class_mean[c].view(1, -1)
            class_scores[:, c] = -torch.matmul(
                torch.matmul(tensor, precision), tensor.t()).diag()

        if_conf = torch.max(class_scores, dim=1)[0]

        pred = ood_logits.argmax(1)
        class_scores = torch.zeros((ood_logits.shape[0], self.num_classes))
        for c in range(self.num_classes):
            tensor = ood_feat.cpu() - self.class_mean[c].view(1, -1)
            class_scores[:, c] = -torch.matmul(
                torch.matmul(tensor, precision), tensor.t()).diag()
        ood_conf = torch.max(class_scores, dim=1)[0]
        conf = np.concatenate([if_conf.cpu().numpy(), ood_conf.cpu().numpy()])
        ood_gt = -1 * np.ones_like(ood_gt)
        label = np.concatenate([np.ones_like(if_conf), ood_gt])
        
        if prec_th is None:
            return ood_metrics(conf, label, tpr_th), None
        else:
            return ood_metrics(conf, label, tpr_th), search_threshold(conf, label, prec_th)
