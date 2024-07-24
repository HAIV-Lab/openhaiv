"""
MSE range estimator
version: 0.0.4
update: 2023-12-20
"""
import torch
from torch import Tensor

from .minmax import MAMinMax


class MSE(MAMinMax):
    """ MSE range estimator.
        During calibration, searching xmin and xmax in the range of [min, max], solved using grid search.
        Quantization range is determined by the minimum MSE calculated by
        $||V - Q(V)||_F^2$, where V is the input tensor, Q is the quantization function.
    
    Args:
        n_bits (int): number of bits
        symmetric (bool): whether to use symmetric quantization
        signed (bool): whether to use signed quantization
        granularity (str): quantization granularity, channel or tensor
        percentile (float): percentile of input tensor to calculate min-max range, Default: 1e-3
        momentum (float): momentum of moving average, Default: -1.
        maxshrink (float): maximum shrinkage ratio, Default: 0.8
        grid (int): number of grid points, Default: 100
        norm (float): norm of MSE, Default: 2.4

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
        norm: float = 2.4,
    ):
        super().__init__(
            n_bits=n_bits, symmetric=symmetric, signed=signed, 
            granularity=granularity, percentile=percentile, momentum=momentum)
        self.maxshrink = maxshrink
        self.grid = grid
        self.norm = norm
    
    def simulate(self, x, scale, zero, qmin, qmax, flag):
        """ simulate quantization
        Args:
            x (torch.Tensor): input tensor
            scale (torch.Tensor): quantization scale
            zero (torch.Tensor): quantization zero point
            qmin (torch.Tensor): quantization minimum value
            qmax (torch.Tensor): quantization maximum value
            flag (str): flag for weight or activation
        Returns:
            torch.Tensor: simulated quantized tensor
        """
        shape = [1] * len(x.shape)
        if flag == 'activation':
            shape[1] = -1
        else:
            shape[0] = -1

        scale = scale.view(*shape)
        zero = zero.view(*shape)

        x = (x/scale - zero).round_().clamp_(qmin, qmax)  # quantize
        x = (x + zero).mul_(scale)  # dequantize
        return x
    
    @staticmethod
    def measure(x, x_sim, **kwargs):
        """ measure the error between x and x_sim
        Args:
            x (torch.Tensor): input tensor
            x_sim (torch.Tensor): simulated tensor
        """
        norm = kwargs['norm']
        return (x - x_sim).abs_().pow_(norm)
    
    def grid_search(self, x, xmin, xmax, flag):
        """ grid search for suitable scale and zero for minimun MSE
        Args:
            x (torch.Tensor): input tensor
            xmin (torch.Tensor): minimum value of input tensor
            xmax (torch.Tensor): maximum value of input tensor
            flag (str): flag for weight or activation
        """
        best = torch.full(xmin.shape, float('inf')).to(xmin.device)
        scale = torch.zeros_like(best)
        zero = torch.zeros_like(best)

        for i in range(int(self.maxshrink * self.grid) + 1):
            p = 1. - i / self.grid
            _xmin = xmin * p
            _xmax = xmax * p

            _scale, _zero, _qmin, _qmax = self.quantize(_xmin, _xmax)
            x_sim = self.simulate(x, _scale, _zero, _qmin, _qmax, flag)

            if self.granularity in ['L', 'Layer', 'layer']:
                err = self.measure(x, x_sim, norm=self.norm).sum()
                if err < best:
                    best, scale, zero = err, _scale, _zero

            elif self.granularity in ['C', 'Channel', 'channel']:
                if flag == 'activation':
                    _x = x.transpose(0, 1).flatten(1)
                    x_sim = x_sim.transpose(0, 1).flatten(1)
                else:
                    _x = x.flatten(1)
                    x_sim = x_sim.flatten(1)

                err = self.measure(_x, x_sim, norm=self.norm).sum(dim=1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp], scale[tmp], zero[tmp] = err[tmp], _scale[tmp], _zero[tmp]

        return scale, zero, _qmin, _qmax
    
    def __call__(self, x: Tensor, flag: str):
        """ calculate the range of input tensor.
        Args:
            x (torch.Tensor): input tensor,
                shape (C, ...) for weight
                shape (N, C, ...) for activation
            flag (str): flag for weight or activation
        """
        xmin, xmax = self.range(x, flag)

        return self.grid_search(x, xmin, xmax, flag)
