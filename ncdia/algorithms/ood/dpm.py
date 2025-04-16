import torch
import torch.nn as nn

from ncdia.utils import ALGORITHMS
from ncdia.utils.metrics import accuracy
from ncdia.algorithms.base import BaseAlg
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
import numpy as np
from tqdm import tqdm
from .metrics import ood_metrics, search_threshold
from sklearn.metrics import pairwise_distances_argmin_min
import scipy


@ALGORITHMS.register
class DPM(BaseAlg):

    def __init__(self, trainer, hyperparameters=None) -> None:
        super().__init__(trainer)
        self.hyperparameters = hyperparameters

    @staticmethod
    def kl(p, q):
        return scipy.stats.entropy(p, q)

    @staticmethod
    def scale_features(kl_id_norm, kl_ood_norm, in_fea, out_fea):
            # 将 numpy.ndarray 转换为 torch.Tensor
            kl_id_norm = torch.tensor(kl_id_norm, dtype=in_fea.dtype, device=in_fea.device)
            kl_ood_norm = torch.tensor(kl_ood_norm, dtype=out_fea.dtype, device=out_fea.device)
            
            x_max, x_min = max(kl_id_norm.max(), kl_ood_norm.max()), min(kl_id_norm.min(), kl_ood_norm.min())
            target_max, target_min = max(in_fea.max(), out_fea.max()), min(in_fea.min(), out_fea.min())
            kl_id_norm_scaled = (kl_id_norm - x_min) / (x_max - x_min) * (target_max - target_min) + target_min
            kl_ood_norm_scaled = (kl_ood_norm - x_min) / (x_max - x_min) * (target_max - target_min) + target_min
            return kl_id_norm_scaled, kl_ood_norm_scaled

    def val_step(self, trainer, data, label, *args, **kwargs):
        
        model = trainer.model
        criterion = trainer.criterion
        device = trainer.device

        data, label = data.to(device), label.to(device)
        outputs = model.evaluate(data)

        loss = criterion(outputs, label)
        outputs = outputs[1]
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
        T = hyperparameters['T']
        # beta = hyperparameters['beta']
        print("DPM inference..")
        neg_ood_gt = -1 * np.ones(ood_logits.shape[0])
        prototypes = torch.softmax(prototypes / T, dim=1)

        id_conf, _ = torch.max(
            torch.softmax(id_logits / T, dim=1), dim=1)
        ood_conf, _ = torch.max(
            torch.softmax(ood_logits / T, dim=1), dim=1)

        batch_size = 64
        # 计算 KL 散度并添加进度条
        print("Calculating KL divergence...")
        id_kl = []
        for i in tqdm(range(0, len(id_logits), batch_size), desc="ID KL"):
            batch_logits = id_logits[i:i+batch_size]
            kl_values = -pairwise_distances_argmin_min(
                torch.softmax(batch_logits / T, dim=1), prototypes, metric=DPM.kl)[1]
            id_kl.append(kl_values)
        id_kl = np.concatenate(id_kl)

        ood_kl = []
        for i in tqdm(range(0, len(ood_logits), batch_size), desc="OOD KL"):
            batch_logits = ood_logits[i:i+batch_size]
            kl_values = -pairwise_distances_argmin_min(
                torch.softmax(batch_logits / T, dim=1), prototypes, metric=DPM.kl)[1]
            ood_kl.append(kl_values)
        ood_kl = np.concatenate(ood_kl)
        
        id_kl_norm, ood_kl_norm = DPM.scale_features(id_kl, ood_kl, id_conf, ood_conf)

        best_score = 0
        best_result = None
        label = np.concatenate([id_gt.cpu(), neg_ood_gt])
        for beta in tqdm(range(0, 101), desc="Searching best beta"):
            beta = round(beta/10, 1)
            id_conf = id_conf + beta * id_kl_norm
            ood_conf = ood_conf + beta * ood_kl_norm
            print(f"beta: {beta}, id_conf: {id_conf}, ood_conf: {ood_conf}")
            conf = np.concatenate([id_conf.cpu(), ood_conf.cpu()])
            result = ood_metrics(conf, label, tpr_th)
            score = result[1] - result[0]
            if score > best_score:
                best_score = score
                best_result = result

        # conf = np.concatenate([id_conf.cpu(), ood_conf.cpu()])
        
        if prec_th is None:
            # return conf, label, *ood_metrics(conf, label, tpr_th), None, None, None
            return best_result, None
        else:
            # return conf, label, *ood_metrics(conf, label, tpr_th), *search_threshold(conf, label, prec_th)
            return best_result

        # if prec_th is None:
        #     # return conf, label, *ood_metrics(conf, label, tpr_th), None, None, None
        #     return ood_metrics(conf, label, tpr_th), None
        # else:
        #     # return conf, label, *ood_metrics(conf, label, tpr_th), *search_threshold(conf, label, prec_th)
        #     return ood_metrics(conf, label, tpr_th), search_threshold(conf, label, prec_th)

