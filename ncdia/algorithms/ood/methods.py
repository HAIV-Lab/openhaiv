import torch
import torch.nn.functional as F

import numpy as np
from numpy.linalg import norm

from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance

from .metrics import ood_metrics


def msp(id_gt, id_logits, ood_gt, ood_logits) -> tuple:
    """Maximum Softmax Probability (MSP) method for OOD detection.

    A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks
    https://arxiv.org/abs/1610.02136

    Args:
        id_gt (torch.Tensor): ID ground truth labels. Shape (N,).
        id_logits (torch.Tensor): ID logits. Shape (N, C).
        ood_gt (torch.Tensor): OOD ground truth labels. Shape (M,).
        ood_logits (torch.Tensor): OOD logits. Shape (M, C).

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

    id_conf, _ = torch.max(
        torch.softmax(id_logits, dim=1), dim=1)
    ood_conf, _ = torch.max(
        torch.softmax(ood_logits, dim=1), dim=1)
    
    conf = np.concatenate([id_conf.cpu(), ood_conf.cpu()])
    label = np.concatenate([id_gt.cpu(), ood_gt])

    return ood_metrics(conf, label)


def mcm(id_gt, id_logits, ood_gt, ood_logits, T: int = 2) -> tuple:
    """Maximum Concept Matching (MCM) method for OOD detection.

    Delving into Out-of-Distribution Detection with Vision-Language Representations
    https://openreview.net/forum?id=KnCS9390Va

    Args:
        id_gt (torch.Tensor): ID ground truth labels. Shape (N,).
        id_logits (torch.Tensor): ID logits. Shape (N, C).
        ood_gt (torch.Tensor): OOD ground truth labels. Shape (M,).
        ood_logits (torch.Tensor): OOD logits. Shape (M, C).
        T (int): Temperature for softmax.

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

    id_conf, _ = torch.max(
        torch.softmax(id_logits / T, dim=1), dim=1)
    ood_conf, _ = torch.max(
        torch.softmax(ood_logits / T, dim=1), dim=1)

    conf = np.concatenate([id_conf.cpu(), ood_conf.cpu()])
    label = np.concatenate([id_gt.cpu(), ood_gt])
    
    return ood_metrics(conf, label)


def max_logit(id_gt, id_logits, ood_gt, ood_logits) -> tuple:
    """Maximum Logit (MaxLogit) method for OOD detection.

    Scaling Out-of-Distribution Detection for Real-World Settings
    https://arxiv.org/abs/1911.11132
    
    Args:
        id_gt (torch.Tensor): ID ground truth labels. Shape (N,).
        id_logits (torch.Tensor): ID logits. Shape (N, C).
        ood_gt (torch.Tensor): OOD ground truth labels. Shape (M,).
        ood_logits (torch.Tensor): OOD logits. Shape (M, C).

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
    
    return ood_metrics(conf, label)


def energy(id_gt, id_logits, ood_gt, ood_logits) -> tuple:
    """Energy-based method for OOD detection.

    Energy-based Out-of-distribution Detection
    https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html

    Args:
        id_gt (torch.Tensor): ID ground truth labels. Shape (N,).
        id_logits (torch.Tensor): ID logits. Shape (N, C).
        ood_gt (torch.Tensor): OOD ground truth labels. Shape (M,).
        ood_logits (torch.Tensor): OOD logits. Shape (M, C).

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
    
    return ood_metrics(conf, label)


def vim(
        id_gt, id_logits, id_feat, 
        ood_gt, ood_logits, ood_feat, 
        train_logits, train_feat
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
    NS = np.ascontiguousarray(
        (eigen_vectors.T[np.argsort(eig_vals * -1)[D:]]).T)
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
    
    return ood_metrics(conf, label)


def dml(id_gt, id_feat, ood_gt, ood_feat, fc_weight) -> tuple:
    """Decoupled MaxLogit (DML) method for OOD detection.

    Decoupling MaxLogit for Out-of-Distribution Detection
    https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Decoupling_MaxLogit_for_Out-of-Distribution_Detection_CVPR_2023_paper

    Args:
        id_gt (torch.Tensor): ID ground truth labels. Shape (N,).
        id_feat (torch.Tensor): ID features. Shape (N, D).
        ood_gt (torch.Tensor): OOD ground truth labels. Shape (M,).
        ood_feat (torch.Tensor): OOD features. Shape (M, D).
        fc_weight (torch.Tensor): FC layer weight. Shape (C, D).

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

    w = fc_weight.detach().clone()
    w = F.normalize(w, p=2, dim=1).cpu()
    # TODO: check if this is correct
    w = w[::2,]  # savc使用的是两倍类别数的fc层

    id_cosine = F.normalize(id_feat, p=2, dim=1) @ w.T
    id_mcos, _ = torch.max(id_cosine, dim=1, keepdim=True)
    id_norm = torch.norm(id_feat, dim=1)
    id_conf = id_mcos + 0.002 * id_norm.unsqueeze(1)

    ood_cosine = F.normalize(ood_feat, p=2, dim=1) @ w.T
    ood_mcos, _ = torch.max(ood_cosine, dim=1, keepdim=True)
    ood_norm = torch.norm(ood_feat, dim=1)
    ood_conf = ood_mcos + 0.002 * ood_norm.unsqueeze(1)

    conf = np.concatenate([id_conf.cpu(), ood_conf.cpu()])
    label = np.concatenate([id_gt.cpu(), ood_gt])
    
    return ood_metrics(conf, label)


def dmlp(
        id_gt, id_logits, id_feat, 
        ood_gt, ood_logits, ood_feat, 
        fc_weight, prototype
    ) -> tuple:
    """Decoupled MaxLogit+ (DML+) method for OOD detection.

    Decoupling MaxLogit for Out-of-Distribution Detection
    https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Decoupling_MaxLogit_for_Out-of-Distribution_Detection_CVPR_2023_paper

    Args:
        id_gt (torch.Tensor): ID ground truth labels. Shape (N,).
        id_logits (torch.Tensor): ID logits. Shape (N, C).
        id_feat (torch.Tensor): ID features. Shape (N, D).
        ood_gt (torch.Tensor): OOD ground truth labels. Shape (M,).
        ood_logits (torch.Tensor): OOD logits. Shape (M, C).
        ood_feat (torch.Tensor): OOD features. Shape (M, D).
        fc_weight (torch.Tensor): FC layer weight. Shape (D, C).
        prototype (torch.Tensor): Prototype. Shape (D, C).

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

    w = fc_weight.clone().detach()
    w = F.normalize(w, p=2, dim=1).cpu()
    # TODO: check if this is correct
    w = w[::2,]  # savc使用的是两倍类别数的fc层

    id_cosine = F.normalize(id_feat, p=2, dim=1) @ w.T
    id_mcos, _ = torch.max(id_cosine, dim=1, keepdim=True)
    id_norm = torch.norm(id_feat, dim=1)
    id_conf = id_mcos + 0.002 * id_norm.unsqueeze(1)

    ood_cosine = F.normalize(ood_feat, p=2, dim=1) @ w.T
    ood_mcos, _ = torch.max(ood_cosine, dim=1, keepdim=True)
    ood_norm = torch.norm(ood_feat, dim=1)
    ood_conf = ood_mcos + 0.002 * ood_norm.unsqueeze(1)

    prototype = F.normalize(prototype, p=2, dim=1)

    _id_conf = F.normalize(id_logits, p=2, dim=1) @ prototype.T
    _id_conf, _ = torch.max(_id_conf, dim=1)

    _ood_conf = F.normalize(ood_logits, p=2, dim=1) @ prototype.T
    _ood_conf, _ = torch.max(_ood_conf, dim=1, keepdim=True)
    
    id_conf = id_conf.cpu() + 40 * _id_conf.cpu()
    ood_conf = ood_conf.cpu() + 40 * _ood_conf.cpu()
    
    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([id_gt.cpu(), ood_gt])
    
    return ood_metrics(conf, label)


def prot(id_gt, id_logits, ood_gt, ood_logits, prototypes) -> tuple:
    """Prototype-based (Prot) method for OOD detection.

    Args:
        id_gt (torch.Tensor): ID ground truth labels, shape (N,).
        id_logits (list of torch.Tensor): ID logits, containing shape (N, C).
        ood_gt (torch.Tensor): OOD ground truth labels, shape (M,).
        ood_logits (list of torch.Tensor): OOD logits, containing shape (M, C).
        prototypes (list of torch.Tensor): Prototypes, containing shape (D, C).

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

    L = len(prototypes)
    id_conf, ood_conf = 0, 0

    for i in range(L):
        prototypes[i] = F.normalize(prototypes[i], p=2, dim=1)

        _id_conf = F.normalize(id_logits[i], p=2, dim=1) @ prototypes[i].T
        _id_conf, _ = torch.max(_id_conf, dim=1)
        id_conf += _id_conf.cpu()

        _ood_conf = F.normalize(ood_logits[i], p=2, dim=1) @ prototypes[i].T
        _ood_conf, _ = torch.max(_ood_conf, dim=1)
        ood_conf += _ood_conf.cpu()

    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([id_gt.cpu(), ood_gt])
    
    return ood_metrics(conf, label)
