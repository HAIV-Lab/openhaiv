"""
Pack tensors
version: 0.0.1
update: 2024-1-4
"""
import torch
from torch import Tensor
import numpy as np


def tpack(x: Tensor, n_bits: int, signed: bool):
    """ pack tensor to uint8
    Args:
        x: input tensor
        n_bits: number of bits, <= 8
        signed: tensor x is signed or not
    Returns:
        packed tensor, description
    """
    des = [n_bits, 1 if signed else 0, *x.shape]
    des = torch.tensor(des, dtype=torch.int32, device=x.device)

    x = x.to(torch.uint8)
    if signed:
        x = x + 2**(n_bits-1)
    x = x.flatten()

    size = (x.shape[0] * n_bits) // 8
    qx = torch.zeros(size, dtype=torch.uint8, device=x.device)

    qxi, flag = 0, 0
    for i in range(x.shape[0]):
        for j in range(n_bits):
            qx[qxi] |= ((x[i] >> j) & 1) << flag
            flag += 1
            qxi += flag // 8
            flag = flag % 8

    return qx, des


def tunpack(qx: Tensor, des: Tensor):
    """ unpack uint8 tensor
    Args:
        qx: input tensor
        des: description of qx,
            0: n_bits, 1: signed, 2: shape[0], 3: shape[1], ...
    Returns:
        unpacked tensor
    """
    n_bits = des[0].item()
    signed = des[1].item()
    shape = des[2:].tolist()

    size = np.prod(shape)
    x = torch.zeros(size, dtype=torch.uint8, device=qx.device)

    qxi, flag = 0, 0
    for i in range(x.shape[0]):
        for j in range(n_bits):
            x[i] |= ((qx[qxi] >> flag) & 1) << j
            flag += 1
            qxi += flag // 8
            flag = flag % 8

    if signed:
        x = x - 2**(n_bits-1)
        x = x.to(torch.int8)

    return x.view(*shape)
