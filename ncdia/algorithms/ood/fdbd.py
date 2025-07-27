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
    """fDBD

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

    def __init__(self, trainer, distance_as_normalizer=True) -> None:
        super().__init__(trainer)
        self.trainer = trainer
        w, b = self.trainer.model.network.fc.parameters()
        self.hyparameters = {
            "distance_as_normalizer": distance_as_normalizer,
            "w": w,
            "b": b,
        }

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
        """fDBD

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
        print("fDBD inference..")
        activation_log_concat = train_feat.cpu().numpy()

        train_mean = torch.from_numpy(np.mean(activation_log_concat, axis=0))

        w, b = hyparameters["w"], hyparameters["b"]
        w = w.data.cpu().numpy()
        b = b.data.cpu().numpy()
        num_classes = b.shape[0]

        """
        for i, param in enumerate(hyparameters["fc_parameters"]):
            if i == 0:
                w = param.data.cpu().numpy()
            else:
                b = param.data.cpu().numpy()
                num_classes = b.shape[0]
        """
        denominator_matrix = np.zeros((num_classes, num_classes))
        for p in range(num_classes):
            w_p = w - w[p, :]
            denominator = np.linalg.norm(w_p, axis=1)
            denominator[p] = 1
            denominator_matrix[p, :] = denominator
        denominator_matrix = torch.from_numpy(denominator_matrix)

        values, nn_idx = id_logits.max(1)
        logits_sub = torch.abs(id_logits - values.repeat(id_logits.shape[1], 1).T)
        if hyparameters["distance_as_normalizer"]:
            score = torch.sum(
                logits_sub / denominator_matrix[nn_idx], axis=1
            ) / torch.norm(id_feat - train_mean, dim=1)
        else:
            score = torch.sum(
                logits_sub / denominator_matrix[nn_idx], axis=1
            ) / torch.norm(id_feat, dim=1)

        values, nn_idx = ood_logits.max(1)
        logits_sub = torch.abs(ood_logits - values.repeat(ood_logits.shape[1], 1).T)
        if hyparameters["distance_as_normalizer"]:
            score_ood = torch.sum(
                logits_sub / denominator_matrix[nn_idx], axis=1
            ) / torch.norm(ood_feat - train_mean, dim=1)
        else:
            score_ood = torch.sum(
                logits_sub / denominator_matrix[nn_idx], axis=1
            ) / torch.norm(ood_feat, dim=1)
        conf = np.concatenate([score.cpu(), score_ood.cpu()])
        ood_gt = -1 * np.ones(score_ood.shape[0])
        label = np.concatenate([id_gt.cpu(), ood_gt])

        if prec_th is None:
            # return conf, label, *ood_metrics(conf, label, tpr_th), None, None, None
            return ood_metrics(conf, label, tpr_th), None
            # return get_measures(id_conf, ood_conf, tpr_th), None
        else:
            # return conf, label, *ood_metrics(conf, label, tpr_th), *search_threshold(conf, label, prec_th)
            return ood_metrics(conf, label, tpr_th), search_threshold(
                conf, label, prec_th
            )
