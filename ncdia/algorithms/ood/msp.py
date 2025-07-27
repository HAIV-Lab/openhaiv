import torch
import torch.nn as nn

from ncdia.utils import ALGORITHMS
from ncdia.utils.metrics import accuracy
from ncdia.algorithms.base import BaseAlg
from ncdia.algorithms.supervised.standard import StandardSL
from ncdia.trainers.hooks import AlgHook, QuantifyHook
import numpy as np
from tqdm import tqdm
from .metrics import ood_metrics, search_threshold
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook


@HOOKS.register
class MSPHook(QuantifyHook):
    def __init__(self) -> None:
        super().__init__()

    def after_test(self, trainer) -> None:
        pass


@ALGORITHMS.register
class MSP(StandardSL):
    """Decoupling MaxLogit.

    Containing:
        - train_step(trainer, data, label, *args, **kwargs)
        - val_step(trainer, data, label, *args, **kwargs)
        - test_step(trainer, data, label, *args, **kwargs)

    """

    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        hook = MSPHook()
        trainer.register_hook(hook)
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
        id_feat: torch.Tensor,
        ood_logits: torch.Tensor,
        ood_feat: torch.Tensor,
        train_logits: torch.Tensor = None,
        train_feat: torch.Tensor = None,
        train_gt: torch.Tensor = None,
        tpr_th: float = 0.95,
        prec_th: float = None,
        hyparameters: dict = None,
        id_local_logits=None,
        id_local_feat=None,
        ood_local_logits=None,
        ood_local_feat=None,
        train_local_logits=None,
        train_local_feat=None,
        prototypes=None,
        s_prototypes=None,
        hyperparameters=None,
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
        print("MSP inference..")
        neg_ood_gt = -1 * np.ones(ood_logits.shape[0])

        id_conf, _ = torch.max(torch.softmax(id_logits, dim=1), dim=1)
        ood_conf, _ = torch.max(torch.softmax(ood_logits, dim=1), dim=1)

        conf = np.concatenate([id_conf.cpu(), ood_conf.cpu()])
        label = np.concatenate([id_gt.cpu(), neg_ood_gt])

        if prec_th is None:
            # return conf, label, *ood_metrics(conf, label, tpr_th), None, None, None
            return ood_metrics(conf, label, tpr_th), None
            # return get_measures(id_conf, ood_conf, tpr_th), None
        else:
            # return conf, label, *ood_metrics(conf, label, tpr_th), *search_threshold(conf, label, prec_th)
            return ood_metrics(conf, label, tpr_th), search_threshold(
                conf, label, prec_th
            )
