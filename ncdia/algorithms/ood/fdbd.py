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
class fDBD(BaseAlg):

    def __init__(self, trainer, distance_as_normalizer=True) -> None:
        super().__init__(trainer)
        self.trainer = trainer
        self.hyperparameters = {
            "distance_as_normalizer": distance_as_normalizer,
            "fc_parameters": self.trainer.model.fc.parameters()
        }

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
        print("fDBD inference..")
        activation_log_concat = train_feat.cpu().numpy()

        train_mean = torch.from_numpy(
                np.mean(activation_log_concat, axis=0))
        
        for i, param in enumerate(hyperparameters["fc_parameters"]):
            if i == 0:
                w = param.data.cpu().numpy()
            else:
                b = param.data.cpu().numpy()
                num_classes = b.shape[0]
                
        denominator_matrix = np.zeros((num_classes, num_classes))
        for p in range(num_classes):
                w_p = w - w[p, :]
                denominator = np.linalg.norm(w_p, axis=1)
                denominator[p] = 1
                denominator_matrix[p, :] = denominator
        denominator_matrix = torch.from_numpy(denominator_matrix)

        values, nn_idx = id_logits.max(1)
        logits_sub = torch.abs(id_logits - values.repeat(id_logits.shape[1], 1).T)
        if hyperparameters["distance_as_normalizer"]:
            score = torch.sum(logits_sub / denominator_matrix[nn_idx],
                              axis=1) / torch.norm(id_feat - train_mean,
                                                   dim=1)
        else:
            score = torch.sum(logits_sub / denominator_matrix[nn_idx],
                              axis=1) / torch.norm(id_feat, dim=1)
            
        values, nn_idx = ood_logits.max(1)
        logits_sub = torch.abs(ood_logits - values.repeat(ood_logits.shape[1], 1).T)
        if hyperparameters["distance_as_normalizer"]:
            score_ood = torch.sum(logits_sub / denominator_matrix[nn_idx],
                              axis=1) / torch.norm(ood_feat - train_mean,
                                                   dim=1)
        else:
            score_ood = torch.sum(logits_sub / denominator_matrix[nn_idx],
                              axis=1) / torch.norm(ood_feat, dim=1)
        conf = np.concatenate([score.cpu(), score_ood.cpu()])
        ood_gt = -1 * np.ones(score_ood.shape[0])
        label = np.concatenate([id_gt.cpu(), ood_gt])
    
        if prec_th is None:
            return ood_metrics(conf, label, tpr_th), None
        else:
            return ood_metrics(conf, label, tpr_th), search_threshold(conf, label, prec_th)
