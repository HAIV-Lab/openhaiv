"""
Cross-entropy range estimator
version: 0.0.1
update: 2023-12-20
"""
from torch import Tensor
import torch.nn.functional as F

from .mse import MSE


class CrossEntropy(MSE):
    """ Cross-entropy range estimator.
        During calibration, searching xmin and xmax in the range of [min, max], solved using grid search.
        Quantization range is determined by the minimum cross-entropy calculated by
        $- \sum_{i=1}^n p_i \log q_i$, where p_i is the softmax of input tensor, q_i is the softmax of quantized tensor.
        Only supports layer-wise granularity and activation quantization.

    Args:
        n_bits (int): number of bits
        symmetric (bool): whether to use symmetric quantization
        signed (bool): whether to use signed quantization
        granularity (str): quantization granularity, channel or tensor
        percentile (float): percentile of input tensor to calculate min-max range, Default: 1e-3
        momentum (float): momentum of moving average, Default: -1.
        maxshrink (float): maximum shrinkage ratio, Default: 0.8
        grid (int): number of grid points, Default: 100

    Returns:
        scale: quantization scale
        zero: quantization zero point
        qmin: quantization minimum value
        qmax: quantization maximum value
    
    """
    def __init__(
        self,
        n_bits: int,
        symmetric: bool,
        signed: bool,
        granularity: str,
        percentile: float = 0.,
        momentum: float = -1.,
        maxshrink: float = 0.8,
        grid: int = 100,
    ):
        super().__init__(
            n_bits=n_bits, symmetric=symmetric, signed=signed, 
            granularity=granularity, percentile=percentile, momentum=momentum,
            maxshrink=maxshrink, grid=grid)
        
    @staticmethod
    def measure(x, x_sim, **kwargs):
        """ measure cross-entropy between x and x_sim
        Args:
            x (torch.Tensor): input tensor
            x_sim (torch.Tensor): simulated tensor
        """
        x = F.softmax(x, dim=-1)
        x_sim = F.softmax(x_sim, dim=-1)
        return F.cross_entropy(x_sim, x, reduction='none')
    
    def __call__(self, x: Tensor, flag: str):
        assert self.granularity in ['L', 'Layer', 'layer'], \
            f'Cross-entropy range estimator only supports layer-wise granularity, got {self.granularity}'
        assert flag == 'activation', \
            f'Cross-entropy range estimator only supports activation quantization, got {flag}'
        return super().__call__(x, flag)
