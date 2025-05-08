import torch
import torch.nn as nn
# import sys
from ncdia.utils import ALGORITHMS
from ncdia.utils.metrics import accuracy
from ncdia.algorithms.base import BaseAlg
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
import numpy as np
from tqdm import tqdm
import scipy
from sklearn.metrics import pairwise_distances_argmin_min
from .metrics import ood_metrics, search_threshold

@ALGORITHMS.register
class KLM(BaseAlg):

    def __init__(self, trainer) -> None:
        super().__init__(trainer)

    @staticmethod
    def kl(p, q):
        return scipy.stats.entropy(p, q)

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
        s_prototypes: torch.Tensor = None,
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
            s_prototypes (torch.Tensor): Softmax Prototypes of train set. Shape (C, C).
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
        print("KLM inference..")
        neg_ood_gt = -1 * np.ones(ood_logits.shape[0])

        id_conf = -pairwise_distances_argmin_min(
                    torch.softmax(id_logits, dim=1), s_prototypes, metric=KLM.kl)[1]
        ood_conf = -pairwise_distances_argmin_min(
                    torch.softmax(ood_logits, dim=1), s_prototypes, metric=KLM.kl)[1]
        
        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt.cpu(), neg_ood_gt])

        if prec_th is None:
            # return conf, label, *ood_metrics(conf, label, tpr_th), None, None, None
            return ood_metrics(conf, label, tpr_th), None
        else:
            # return conf, label, *ood_metrics(conf, label, tpr_th), *search_threshold(conf, label, prec_th)
            return ood_metrics(conf, label, tpr_th), search_threshold(conf, label, prec_th)
