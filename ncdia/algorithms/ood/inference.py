import torch
import torch.nn.functional as F

import numpy as np
from numpy.linalg import norm
import math

import scipy
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import pairwise_distances_argmin_min

from .metrics import ood_metrics, search_threshold


def mls_inf(
    logits,
) -> tuple:
    conf, _ = torch.max(logits, dim=1)
    return conf.cpu()


def msp_inf(
    logits,
) -> tuple:
    conf, _ = torch.max(torch.softmax(logits, dim=1), dim=1)
    return conf.cpu()


def energy_inf(
    logits,
) -> tuple:

    conf = logsumexp(logits.cpu(), axis=-1)
    return conf.cpu()


def vim_inf(
    logits,
    feat,
    train_logits,
    train_feat,
) -> tuple:

    D = train_feat.shape[1] // 2
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(train_feat.cpu())
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[D:]]).T)
    vlogit_id_train = norm(np.matmul(train_feat.cpu(), NS), axis=-1)
    alpha = train_logits.max(axis=-1)[0].mean() / vlogit_id_train.mean()

    energy = logsumexp(logits.cpu(), axis=-1)
    vlogit = norm(np.matmul(feat.numpy(), NS), axis=-1) * alpha.cpu().numpy()

    conf = -vlogit + energy
    return conf.cpu()
