"""
ACIQ range estimator
version: 0.0.1
update: 2023-12-21
"""
import warnings

import torch
from torch import Tensor

from .minmax import MinMax


class ACIQ(MinMax):
    """ Analytical Clipping for Integer Quantization (ACIQ)

    Post training 4-bit quantization of convolutional networks for rapid-deployment
    https://arxiv.org/abs/1810.05723

    Args:
        n_bits (int): number of bits
        symmetric (bool): whether to use symmetric quantization
        signed (bool): whether to use signed quantization
        granularity (str): quantization granularity, channel or tensor
        fuse_relu (bool): whether to fuse ReLU. Default: False

    Returns:
        scale: quantization scale
        zero: quantization zero point
        qmin: quantization minimum value
        qmax: quantization maximum value

    """
    # mapping from n_bits to :math:`C(M)`
    C = [ 1.86,  2.83,  3.90,  5.03, 
          6.20,  7.41,  8.65,  9.90,
         11.16, 12.44, 13.73, 15.02,
         16.33, 17.64, 18.95, 20.27]
    
    # mapping from n_bits to :math:`C_F(M)` for fused ReLU
    Cf = [ 2.83,  3.90,  5.03,  6.20,
           7.41,  8.65,  9.90, 11.16,
          12.44, 13.73, 15.02, 16.33,
          17.64, 18.95, 20.27, 21.59]

    def __init__(
        self,
        n_bits: int,
        symmetric: bool,
        signed: bool,
        granularity: str,
        fuse_relu: bool = False,
        **kwargs
    ):
        super().__init__(
            n_bits=n_bits, symmetric=symmetric, signed=signed, 
            granularity=granularity, percentile=0.)
        self.n_bits = n_bits
        self.fuse_relu = fuse_relu

        if self.fuse_relu and self.symmetric and self.signed:
            msg = 'ACIQ on fused ReLU is not suitable for signed symmetric quantization. ' + \
                  'Use unsigned symmetric or asymmetric quantization instead.'
            warnings.warn(msg)

        self.num = 0
        self.mu_sum = 0.
        self.lam_sum = 0.

    def update(self, x: Tensor, flag: str):
        """ statistics of :math:`\mu` and :math:`\lambda` of laplace distribution.
        Args:
            x (torch.Tensor): input tensor
            flag (str): flag of activation or weight
        Returns:
            torch.Tensor: updated :math:`\mu`
            torch.Tensor: updated :math:`\lambda`
        """
        if self.granularity in ['L', 'Layer', 'layer']:
            self.num += x.numel()
            self.mu_sum += x.sum()
            mu = self.mu_sum / self.num
            self.lam_sum += (x - mu).abs().sum()
            lam = self.lam_sum / self.num
        
        elif self.granularity in ['C', 'Channel', 'channel']:
            if flag == 'activation':
                x = x.transpose(0, 1)
            x = x.flatten(1)
            self.num += x.size(1)
            self.mu_sum += x.sum(1)
            mu = self.mu_sum / self.num
            self.lam_sum += (x - mu.unsqueeze(1)).abs().sum(1)
            lam = self.lam_sum / self.num
        
        else:
            raise NotImplementedError(f'Granularity {self.granularity} not implemented.')
        
        return mu, lam
    
    def range(self, x: Tensor, flag: str):
        """ estimate quantization range.
        Args:
            x (torch.Tensor): input tensor
            flag (str): flag of activation or weight
        Returns:
            torch.Tensor: minimum value
            torch.Tensor: maximum value
        """
        mu, lam = self.update(x, flag)
        
        if not self.fuse_relu:
            alpha = self.C[self.n_bits - 1] * lam
            xmin = mu - alpha
            xmax = mu + alpha

        else:
            alpha = self.Cf[self.n_bits - 1] * lam
            xmin = torch.zeros_like(mu)
            xmax = torch.max(mu, xmin) + alpha

        return xmin, xmax
