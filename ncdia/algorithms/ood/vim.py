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
    def eval(
        id_gt: torch.Tensor,
        id_logits: torch.Tensor,
        id_feat: torch.Tensor,
        ood_logits: torch.Tensor,
        ood_feat: torch.Tensor,
        train_gt: torch.Tensor,
        train_logits: torch.Tensor,
        train_feat: torch.Tensor,
        tpr_th: float = 0.95,
        prec_th: float = None,
    ):
        print("VIM inference..")
        ood_gt = -1 * np.ones(ood_logits.shape[0])

        D = train_feat.shape[1] // 2
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(train_feat.cpu())
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[D:]]).T)
        vlogit_id_train = norm(np.matmul(train_feat.cpu(), NS), axis=-1)
        alpha = train_logits.max(axis=-1)[0].mean() / vlogit_id_train.mean()

        id_energy = logsumexp(id_logits.cpu(), axis=-1)
        ood_energy = logsumexp(ood_logits.cpu(), axis=-1)
        id_vlogit = norm(np.matmul(id_feat.numpy(), NS), axis=-1) * alpha.cpu().numpy()
        ood_vlogit = norm(np.matmul(ood_feat.numpy(), NS), axis=-1) * alpha.cpu().numpy()

        id_conf = -id_vlogit + id_energy
        ood_conf = -ood_vlogit + ood_energy

        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt.cpu(), ood_gt])

        if prec_th is None:
            return ood_metrics(conf, label, tpr_th), None
        else:
            return ood_metrics(conf, label, tpr_th), search_threshold(conf, label, prec_th)
