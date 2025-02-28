import torch
import torch.nn.functional as F

import numpy as np
from numpy.linalg import norm

from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance

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