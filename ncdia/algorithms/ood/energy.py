import torch
import torch.nn as nn

from ncdia.utils import ALGORITHMS
from ncdia.utils.metrics import accuracy
from ncdia.algorithms.base import BaseAlg
from ncdia.algorithms.supervised.standard import StandardSL
from ncdia.trainers.hooks import AlgHook, QuantifyHook
import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
from .metrics import ood_metrics, search_threshold
from ncdia.trainers.hooks import AlgHook

@ALGORITHMS.register
class ENERGY(StandardSL):

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
    def eval(
        id_gt: torch.Tensor,
        id_logits: torch.Tensor,
        ood_logits: torch.Tensor,
        tpr_th: float = 0.95,
        prec_th: float = None,
    ):
        """
        Args:
            id_gt (torch.Tensor): Ground truth labels for in-distribution samples. Shape (N,).
            id_logits (torch.Tensor): ID logits. Shape (N, C).
            ood_logits (torch.Tensor): OOD logits. Shape (M, C).
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
        print("MSP inference..")
        neg_ood_gt = -1 * np.ones(ood_logits.shape[0])

        id_conf = logsumexp(id_logits.cpu(), axis=-1)
        ood_conf = logsumexp(ood_logits.cpu(), axis=-1)

        conf = np.concatenate([id_conf.cpu(), ood_conf.cpu()])
        label = np.concatenate([id_gt.cpu(), neg_ood_gt])

        if prec_th is None:
            # return conf, label, *ood_metrics(conf, label, tpr_th), None, None, None
            return ood_metrics(conf, label, tpr_th), None
            # return get_measures(id_conf, ood_conf, tpr_th), None
        else:
            # return conf, label, *ood_metrics(conf, label, tpr_th), *search_threshold(conf, label, prec_th)
            return ood_metrics(conf, label, tpr_th), search_threshold(conf, label, prec_th)
