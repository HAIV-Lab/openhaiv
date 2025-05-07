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
    """MDS

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

    @staticmethod
    def eval(id_gt: torch.Tensor ,id_logits: torch.Tensor, id_feat: torch.Tensor, 
            ood_logits: torch.Tensor, ood_feat: torch.Tensor, 
            train_logits: torch.Tensor = None, train_feat: torch.Tensor = None, train_gt: torch.Tensor = None,
            tpr_th: float = 0.95, prec_th: float = None, hyparameters: dict = None):
        """MDS
        """
        class_mean = []
        centered_data = []
        num_classes = train_gt.max().item() + 1
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

        class_scores = torch.zeros((id_logits.shape[0], num_classes))
        for c in range(num_classes):
            tensor = id_feat.cpu() - class_mean[c].view(1, -1)
            class_scores[:, c] = -torch.matmul(
                torch.matmul(tensor, precision), tensor.t()).diag()

        if_conf = torch.max(class_scores, dim=1)[0]

        pred = ood_logits.argmax(1)
        class_scores = torch.zeros((ood_logits.shape[0], num_classes))
        for c in range(num_classes):
            tensor = ood_feat.cpu() - class_mean[c].view(1, -1)
            class_scores[:, c] = -torch.matmul(
                torch.matmul(tensor, precision), tensor.t()).diag()
        ood_conf = torch.max(class_scores, dim=1)[0]
        conf = np.concatenate([if_conf.cpu().numpy(), ood_conf.cpu().numpy()])
        
        neg_ood_gt = -1 * np.ones(ood_logits.shape[0])
        
        label = np.concatenate([np.ones_like(if_conf), neg_ood_gt])
        
        if prec_th is None:
            return conf, label, *ood_metrics(conf, label, tpr_th), None, None, None
        else:
            return conf, label, *ood_metrics(conf, label, tpr_th), *search_threshold(conf, label, prec_th)
