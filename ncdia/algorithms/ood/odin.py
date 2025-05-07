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
class ODIN(BaseAlg):
    """ODIN

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
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.hyparameters = None
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
            train_logits: torch.Tensor = None, train_feat: torch.Tensor = None, train_gt: torch.Tensor = None, 
            tpr_th: float = 0.95, prec_th: float = None, hyparameters: dict = None):
        """ODIN
        """
        neg_ood_gt = -1 * np.ones(ood_logits.shape[0])

        id_conf, _ = torch.max(
            torch.softmax(id_logits, dim=1), dim=1)
        ood_conf, _ = torch.max(
            torch.softmax(ood_logits, dim=1), dim=1)
        
        conf = np.concatenate([id_conf.cpu(), ood_conf.cpu()])
        label = np.concatenate([id_conf, neg_ood_gt])
        # label = np.concatenate([np.ones_like(id_conf), neg_ood_gt])
        # neg_label = np.concatenate([np.ones_like(id_conf), neg_ood_gt])

        
        if prec_th is None:
            return conf, label, *ood_metrics(conf, label, tpr_th), None, None, None
        else:
            return conf, label, *ood_metrics(conf, label, tpr_th), *search_threshold(conf, label, prec_th)
