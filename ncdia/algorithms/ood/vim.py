import torch
import torch.nn as nn

from ncdia.utils import ALGORITHMS
from ncdia.utils.metrics import accuracy
from ncdia.algorithms.base import BaseAlg
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
import numpy as np
from numpy.linalg import norm, pinv
from sklearn.covariance import EmpiricalCovariance
from scipy.special import logsumexp
from tqdm import tqdm
from .metrics import ood_metrics, search_threshold

@ALGORITHMS.register
class VIM(BaseAlg):
    """Virtual-Logit Matching (ViM) method for OOD detection.

    ViM: Out-of-Distribution With Virtual-Logit Matching
    https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html

    Args:
        id_gt (torch.Tensor): ID ground truth labels. Shape (N,).
        id_logits (torch.Tensor): ID logits. Shape (N, C).
        id_feat (torch.Tensor): ID features. Shape (N, D).
        ood_gt (torch.Tensor): OOD ground truth labels. Shape (M,).
        ood_logits (torch.Tensor): OOD logits. Shape (M, C).
        ood_feat (torch.Tensor): OOD features. Shape (M, D).
        train_logits (torch.Tensor): Training logits. Shape (K, C).
        train_feat (torch.Tensor): Training features. Shape (K, D).
        tpr_th (float): True positive rate threshold to compute
            false positive rate. Default is 0.95.
        prec_th (float | None): Precision threshold for searching threshold.
            If None, not searching for threshold. Default is None.

    Returns:
        fpr (float): False positive rate.
        auroc (float): Area under the ROC curve.
        aupr_in (float): Area under the precision-recall curve 
            for in-distribution samples.
        aupr_out (float): Area under the precision-recall curve
            for out-of-distribution
    """
    def __init__(self, trainer, dim) -> None:
        super().__init__(trainer)
        self.trainer = trainer
        # use a dict to store hyperparameters
        self.hyparameters = {"dim": dim, 'w': self.trainer.model.network.fc.weight.data.cpu().numpy(), 'b': self.trainer.model.network.fc.bias.data.cpu().numpy()}

    def val_step(self, trainer, data, label, *args, **kwargs):
        """Validation step for Decoupling MaxLogit.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Validation results. Contains the following:
                - "loss": Loss value.
                - "acc": Accuracy value.
        """
        model = trainer.model
        device = trainer.device

        data, label = data.to(device), label.to(device)
        outputs = model(data)

        criterion = nn.CrossEntropyLoss()

        loss = criterion(outputs, label)
        acc = accuracy(outputs, label)[0]

        return {"loss": loss.item(), "acc": acc.item()}


    def test_step(self, trainer, data, label, *args, **kwargs):
        """Test step for Decoupling MaxLogit.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Test results. Contains the following:
                - "loss": Loss value.
                - "acc": Accuracy value.
        """
        return self.val_step(trainer, data, label, *args, **kwargs)

    @staticmethod
    def eval(id_gt: torch.Tensor ,id_logits: torch.Tensor, id_feat: torch.Tensor, 
            ood_logits: torch.Tensor, ood_feat: torch.Tensor, 
            train_logits: torch.Tensor = None, train_feat: torch.Tensor = None,  train_gt: torch.Tensor = None,
            tpr_th: float = 0.95, prec_th: float = None, hyparameters: dict = None,
            id_local_logits = None, id_local_feat = None, ood_local_logits = None,
            ood_local_feat = None, train_local_logits = None, train_local_feat = None,
            prototypes = None, s_prototypes = None, hyperparameters = None
            ):
        """Decoupled MaxLogit+ (DML+) method for OOD detection.

        Decoupling MaxLogit for Out-of-Distribution Detection
        https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Decoupling_MaxLogit_for_Out-of-Distribution_Detection_CVPR_2023_paper

        Args:
            id_logits (torch.Tensor): ID logits. Shape (N, C).
            id_feat (torch.Tensor): ID features. Shape (N, D).
            ood_logits (torch.Tensor): OOD logits. Shape (M, C).
            ood_feat (torch.Tensor): OOD features. Shape (M, D).
            train_logits (torch.Tensor): Training logits. Shape (K, C).
            train_feat (torch.Tensor): Training features. Shape (K, D).
            tpr_th (float): True positive rate threshold to compute
                false positive rate. Default is 0.95.
            prec_th (float | None): Precision threshold for searching threshold.
                If None, not searching for threshold. Default is None.

        Returns:
            fpr (float): False positive rate.
            auroc (float): Area under the ROC curve.
            aupr_in (float): Area under the precision-recall curve 
                for in-distribution samples.
            aupr_out (float): Area under the precision-recall curve
                for out-of-distribution
        """
        print("VIM inference..")

        w, b = hyparameters['w'], hyparameters['b']
        dim = hyparameters['dim']
        feature_id_train = train_feat.cpu().numpy()
        logit_id_train = feature_id_train @ w.T + b
        u = -np.matmul(pinv(w), b)
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray(
            (eigen_vectors.T[np.argsort(eig_vals * -1)[dim:]]).T)

        vlogit_id_train = norm(np.matmul(feature_id_train - u,
                                            NS),
                                axis=-1)
        alpha = logit_id_train.max(
            axis=-1).mean() / vlogit_id_train.mean()
        print(f'{alpha=:.4f}')
        
        id_feat = id_feat.cpu()
        logit_ood = id_feat @ w.T + b
        _, pred = torch.max(logit_ood, dim=1)
        energy_ood = logsumexp(logit_ood.numpy(), axis=-1)
        vlogit_ood = norm(np.matmul(id_feat.numpy() - u, NS),
                          axis=-1) * alpha
        id_score = -vlogit_ood + energy_ood

        ood_feat = ood_feat.cpu()
        logit_ood = ood_feat @ w.T + b
        _, pred = torch.max(logit_ood, dim=1)
        energy_ood = logsumexp(logit_ood.numpy(), axis=-1)
        vlogit_ood = norm(np.matmul(ood_feat.numpy() - u, NS),
                          axis=-1) * alpha
        ood_score = -vlogit_ood + energy_ood

        conf = np.concatenate([id_score, ood_score])
        neg_ood_gt = -1 * np.ones(ood_logits.shape[0])
        label = np.concatenate([id_gt.cpu(), neg_ood_gt])

        if prec_th is None:
            # return conf, label, *ood_metrics(conf, label, tpr_th), None, None, None
            return ood_metrics(conf, label, tpr_th), None
            # return get_measures(id_conf, ood_conf, tpr_th), None
        else:
            # return conf, label, *ood_metrics(conf, label, tpr_th), *search_threshold(conf, label, prec_th)
            return ood_metrics(conf, label, tpr_th), search_threshold(conf, label, prec_th)