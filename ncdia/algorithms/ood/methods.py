import torch
import torch.nn.functional as F

import numpy as np
from numpy.linalg import norm

from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance

from ncdia.utils import ALGORITHMS
from .metrics import ood_metrics, search_threshold


@ALGORITHMS.register
def msp(
    id_gt,
    id_logits,
    ood_gt,
    ood_logits,
    tpr_th: float = 0.95,
    prec_th: float = None,
    **kwargs,
) -> tuple:
    """Maximum Softmax Probability (MSP) method for OOD detection.

    A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks
    https://arxiv.org/abs/1610.02136

    Args:
        id_gt (torch.Tensor): ID ground truth labels. Shape (N,).
        id_logits (torch.Tensor): ID logits. Shape (N, C).
        ood_gt (torch.Tensor): OOD ground truth labels. Shape (M,).
        ood_logits (torch.Tensor): OOD logits. Shape (M, C).
        tpr_th (float): True positive rate threshold to compute
            false positive rate. Default is 0.95.
        prec_th (float | None): Precision threshold for searching threshold.
            If None, not searching for threshold. Default is None.

    Returns:
        conf (np.ndarray): Confidence scores. Shape (N + M,).
        label (np.ndarray): Label array. Shape (N + M,).
        fpr (float): False positive rate.
        auroc (float): Area under the ROC curve.
        aupr_in (float): Area under the precision-recall curve
            for in-distribution samples.
        aupr_out (float): Area under the precision-recall curve
            for out-of-distribution
        best_th (float): Threshold for OOD detection. If prec_th is None, None.
        prec (float): Precision at the threshold. If prec_th is None, None.
        recall (float): Recall at the threshold. If prec_th is None, None.
    """
    # set the ground truth labels for OOD samples to -1
    # for computing ood metrics
    neg_ood_gt = -1 * np.ones_like(ood_gt)

    id_conf, _ = torch.max(torch.softmax(id_logits, dim=1), dim=1)
    ood_conf, _ = torch.max(torch.softmax(ood_logits, dim=1), dim=1)

    conf = np.concatenate([id_conf.cpu(), ood_conf.cpu()])
    label = np.concatenate([id_gt, ood_gt])
    neg_label = np.concatenate([id_gt, neg_ood_gt])

    if prec_th is None:
        return conf, label, *ood_metrics(conf, neg_label, tpr_th), None, None, None
    else:
        return (
            conf,
            label,
            *ood_metrics(conf, neg_label, tpr_th),
            *search_threshold(conf, neg_label, prec_th),
        )

@ALGORITHMS.register
def mls(
    id_gt,
    id_logits,
    ood_gt,
    ood_logits,
    tpr_th: float = 0.95,
    prec_th: float = None,
    **kwargs,
) -> tuple:
    """Maximum Logit (MaxLogit) method for OOD detection.

    Scaling Out-of-Distribution Detection for Real-World Settings
    https://arxiv.org/abs/1911.11132

    Args:
        id_gt (torch.Tensor): ID ground truth labels. Shape (N,).
        id_logits (torch.Tensor): ID logits. Shape (N, C).
        ood_gt (torch.Tensor): OOD ground truth labels. Shape (M,).
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
    # set the ground truth labels for OOD samples to -1
    # for computing ood metrics
    ood_gt = -1 * np.ones_like(ood_gt)

    id_conf, _ = torch.max(id_logits, dim=1)
    ood_conf, _ = torch.max(ood_logits, dim=1)

    conf = np.concatenate([id_conf.cpu(), ood_conf.cpu()])
    label = np.concatenate([id_gt.cpu(), ood_gt])

    if prec_th is None:
        return ood_metrics(conf, label, tpr_th), None
    else:
        return ood_metrics(conf, label, tpr_th), search_threshold(conf, label, prec_th)


@ALGORITHMS.register
def energy(
    id_gt,
    id_logits,
    ood_gt,
    ood_logits,
    tpr_th: float = 0.95,
    prec_th: float = None,
    **kwargs,
) -> tuple:
    """Energy-based method for OOD detection.

    Energy-based Out-of-distribution Detection
    https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html

    Args:
        id_gt (torch.Tensor): ID ground truth labels. Shape (N,).
        id_logits (torch.Tensor): ID logits. Shape (N, C).
        ood_gt (torch.Tensor): OOD ground truth labels. Shape (M,).
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
    # set the ground truth labels for OOD samples to -1
    # for computing ood metrics
    ood_gt = -1 * np.ones_like(ood_gt)

    id_conf = logsumexp(id_logits.cpu(), axis=-1)
    ood_conf = logsumexp(ood_logits.cpu(), axis=-1)

    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([id_gt.cpu(), ood_gt])

    if prec_th is None:
        return ood_metrics(conf, label, tpr_th), None
    else:
        return ood_metrics(conf, label, tpr_th), search_threshold(conf, label, prec_th)


@ALGORITHMS.register
def vim(
    id_gt,
    id_logits,
    id_feat,
    ood_gt,
    ood_logits,
    ood_feat,
    train_logits,
    train_feat,
    tpr_th: float = 0.95,
    prec_th: float = None,
    **kwargs,
) -> tuple:
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
    # set the ground truth labels for OOD samples to -1
    # for computing ood metrics
    ood_gt = -1 * np.ones_like(ood_gt)

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