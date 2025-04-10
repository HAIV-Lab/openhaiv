import torch
import torch.nn.functional as F

import numpy as np
from numpy.linalg import norm
import math

from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import pairwise_distances_argmin_min

from .metrics import ood_metrics, search_threshold


def msp_inf(
        logits,
) -> tuple:
    conf, _ = torch.max(torch.softmax(logits, dim=1), dim=1)
    return conf.cpu()


def mcm_inf(
        logits, T: int = 2,
) -> tuple:
    conf, _ = torch.max(torch.softmax(logits / T, dim=1), dim=1)

    return conf.cpu()


def glmcm_inf(
        global_logits, 
        local_logits,
        lambda_local: float = 1,
        T: int = 2,
) -> tuple:
    global_conf, _ = torch.max(torch.softmax(global_logits / T, dim=1), dim=1)
    local_conf, _ = torch.max(torch.softmax(local_logits / T, dim=1), dim=(1,2))
    conf, _ = global_conf + lambda_local * local_conf
    return conf.cpu()


def neglabel_inf(
        positive_logits,
        negative_logits,
        T: int = 2,
) -> tuple:
    total_logits = torch.cat((positive_logits, negative_logits), dim=1)
    total_conf, _ = torch.max(torch.softmax(total_logits / T, dim=1), dim=1)
    conf = total_conf[:, :positive_logits.shape[1]]
    return conf.cpu()


def max_logit_inf(
        logits,
) -> tuple:
    conf, _ = torch.max(logits, dim=1)
    return conf.cpu()


def energy_inf(
        logits,
) -> tuple:

    conf = logsumexp(logits.cpu(), axis=-1)
    return conf.cpu()


def vim_inf(
        logits, feat,
        train_logits, train_feat,
) -> tuple:

    D = train_feat.shape[1] // 2
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(train_feat.cpu())
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray(
        (eigen_vectors.T[np.argsort(eig_vals * -1)[D:]]).T)
    vlogit_id_train = norm(np.matmul(train_feat.cpu(), NS), axis=-1)
    alpha = train_logits.max(axis=-1)[0].mean() / vlogit_id_train.mean()

    energy = logsumexp(logits.cpu(), axis=-1)
    vlogit = norm(np.matmul(feat.numpy(), NS), axis=-1) * alpha.cpu().numpy()

    conf = -vlogit + energy
    return conf.cpu()

def dml_inf(
        feat,
        fc_weight: torch.Tensor,
) -> tuple:
    w = fc_weight.detach().clone()
    w = F.normalize(w, p=2, dim=1).cpu()
    # TODO: check if this is correct
    w = w[::2, ]  # savc使用的是两倍类别数的fc层

    cosine = F.normalize(feat, p=2, dim=1) @ w.T
    mcos, _ = torch.max(cosine, dim=1, keepdim=True)
    norm = torch.norm(feat, dim=1)
    conf = mcos + 0.002 * norm.unsqueeze(1)
    return conf.cpu()


def dmlp_inf(
        logits, feat,
        fc_weight, prototype,
) -> tuple:
    w = fc_weight.clone().detach()
    w = F.normalize(w, p=2, dim=1).cpu()
    # TODO: check if this is correct
    w = w[::2, ]  # savc使用的是两倍类别数的fc层

    cosine = F.normalize(feat, p=2, dim=1) @ w.T
    mcos, _ = torch.max(cosine, dim=1, keepdim=True)
    norm = torch.norm(feat, dim=1)
    conf = mcos + 0.002 * norm.unsqueeze(1)

    prototype = F.normalize(prototype, p=2, dim=1)

    _conf = F.normalize(logits, p=2, dim=1) @ prototype.T
    _conf, _ = torch.max(_conf, dim=1)


    conf = conf.cpu() + 40 * _conf.cpu()
    return conf


def prot_inf(
        logits,
        prototypes: list,
) -> tuple:
    L = len(prototypes)
    conf = 0
    # label = 0
    for i in range(L):
        prototypes[i] = F.normalize(prototypes[i], p=2, dim=1)

        _conf = F.normalize(logits[i], p=2, dim=1) @ prototypes[i].T
        _conf, _ = torch.max(_conf, dim=1)
        conf += _conf.cpu()

    return conf

def klm_inf(
        logits, 
        train_logits: np.ndarray,
) -> tuple:
    conf = -pairwise_distances_argmin_min(
        torch.softmax(logits), train_logits, metric='kl')[1]
    
    return conf

def she_inf(
        logits: torch.Tensor,
        features: torch.Tensor,
        train_features: torch.Tensor,
        metric: str = 'inner_product',
) -> tuple:
    # Get predicted class indices
    preds = logits.argmax(dim=1)

    # Compute distances
    if metric == 'inner_product':
        conf = torch.sum(features * train_features[preds], dim=1)
    elif metric == 'euclidean':
        conf = -torch.sqrt(torch.sum((features - train_features[preds])**2, dim=1))
    elif metric == 'cosine':
        conf = torch.cosine_similarity(features, train_features[preds], dim=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return conf

def relation_inf(
        logits: torch.Tensor,
        features: torch.Tensor,
        train_logits: torch.Tensor,
        train_features: torch.Tensor,
        pow: int = 1,
        thres: float = 0.03,
        chunk: int = 50,
) -> tuple:
    
    def kernel(feat, feat_t, logits, logits_t, split=2):
        size = math.ceil(len(feat_t) / split)
        rel_full = []
        for i in range(split):
            feat_t_ = feat_t[i * size:(i + 1) * size]
            logits_t_ = logits_t[i * size:(i + 1) * size]

            with torch.no_grad():
                dot = torch.matmul(feat, feat_t_.transpose(1, 0))
                dot = torch.clamp(dot, min=0.)

                sim = torch.matmul(logits, logits_t_.transpose(1, 0))
                rel = dot * sim

            rel_full.append(rel)

        rel_full = torch.cat(rel_full, dim=-1)
        return rel_full
    
    def get_relation(feat, feat_t, logits, logits_t, pow=1, chunk=50, thres=0.03):
        n = feat.shape[0]
        n_chunk = math.ceil(n / chunk)

        score = []
        for i in range(n_chunk):
            feat_ = feat[i * chunk:(i + 1) * chunk]
            logits_ = logits[i * chunk:(i + 1) * chunk]

            rel = kernel(feat_, feat_t, logits_, logits_t)

            mask = (rel.abs() > thres)
            rel_mask = mask * rel
            edge_sum = (rel_mask.sign() * (rel_mask.abs()**pow)).sum(-1)

            score.append(edge_sum.cpu())

        score = torch.cat(score, dim=0)

        return score
    
    # Normalize features
    features = F.normalize(features, dim=1)
    train_features = F.normalize(train_features, dim=1)

    # Compute relation scores
    conf = get_relation(features, train_features, logits, train_logits, pow=pow, chunk=chunk, thres=thres)


    return conf

def mds_inf(
        logits: torch.Tensor,
        features: torch.Tensor,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        num_classes: int,
) -> tuple:
    """
    MDS inference for OOD detection.

    Args:
        logits (torch.Tensor): Logits of the input samples. Shape (N, C).
        features (torch.Tensor): Features of the input samples. Shape (N, D).
        train_features (torch.Tensor): Features of the training samples. Shape (N_train, D).
        train_labels (torch.Tensor): Ground truth labels of the training samples. Shape (N_train,).
        num_classes (int): Number of classes.

    Returns:
        tuple: (preds, conf)
            preds (torch.Tensor): Predicted class indices. Shape (N,).
            conf (torch.Tensor): Confidence scores based on Mahalanobis distance. Shape (N,).
    """
    # Compute class means from training features
    class_mean = []
    centered_data = []
    for c in range(num_classes):
        class_samples = train_features[train_labels == c]
        class_mean.append(class_samples.mean(0))
        centered_data.append(class_samples - class_samples.mean(0).view(1, -1))

    class_mean = torch.stack(class_mean)  # Shape: [num_classes, feature_dim]

    # Compute covariance matrix and its inverse
    centered_data = torch.cat(centered_data, dim=0)
    group_lasso = EmpiricalCovariance(assume_centered=False)
    group_lasso.fit(centered_data.cpu().numpy().astype(np.float32))
    precision = torch.from_numpy(group_lasso.precision_).float()  # Shape: [feature_dim, feature_dim]

    # Compute Mahalanobis distance for each class
    class_scores = torch.zeros((logits.shape[0], num_classes))
    for c in range(num_classes):
        tensor = features - class_mean[c].view(1, -1)
        class_scores[:, c] = -torch.matmul(
            torch.matmul(tensor, precision), tensor.t()).diag()

    # Get maximum Mahalanobis distance as confidence score
    conf, _ = torch.max(class_scores, dim=1)

    return conf