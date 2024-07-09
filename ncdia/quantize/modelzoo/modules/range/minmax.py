"""
Min-max range estimator.
version: 0.1.1
update: 2023-12-20
"""
import torch
from torch import Tensor


class MinMax():
    """ Min-max range estimator.
        During calibration, the minimum and maximum values of all input batches are recorded.
        The quantization range is determined by the minimum and maximum values.

    Args:
        n_bits (int): number of bits
        symmetric (bool): whether to use symmetric quantization
        signed (bool): whether to use signed quantization
        granularity (str): quantization granularity, channel or tensor
        percentile (float): percentile of input tensor to calculate min-max range, Default: 1e-3

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
        **kwargs
    ):
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.signed = signed
        self.granularity = granularity
        self.percentile = percentile

    def update(self, xmin, xmax):
        """ update min-max range.
        Args:
            xmin (torch.Tensor): minimum value
            xmax (torch.Tensor): maximum value
        Returns:
            torch.Tensor: updated minimum value
            torch.Tensor: updated maximum value
        """
        if 'xmin' not in self.__dict__:
            self.xmin = xmin
            self.xmax = xmax
        else:
            self.xmin = torch.min(self.xmin, xmin)
            self.xmax = torch.max(self.xmax, xmax)

        return self.xmin, self.xmax

    def range(self, x: Tensor, flag: str):
        """ calculate the range of input tensor.
        Args:
            x (torch.Tensor): input tensor,
                shape (C, ...) for weight
                shape (N, C, ...) for activation
            flag (str): flag for weight or activation
        Returns:
            torch.Tensor: minimum value
            torch.Tensor: maximum value
        """
        if self.granularity in ['L', 'Layer', 'layer']:
            x = x.flatten(0)
            if self.percentile == 0.:
                xmin = x.min() if not self.symmetric else torch.tensor(0.).to(x.device)
                xmax = x.max() if not self.symmetric else x.abs().max()
            elif not self.symmetric:
                xmin = x.kthvalue(int(x.numel() * self.percentile) + 1)[0]
                xmax = x.kthvalue(int(x.numel() * (1 - self.percentile)))[0]
            else:
                xmin = torch.tensor(0.).to(x.device)
                xmax = x.abs().kthvalue(int(x.numel() * (1 - self.percentile)))[0]

        elif self.granularity in ['C', 'Channel', 'channel']:
            if flag == 'activation':
                x = x.transpose(0, 1)
            x = x.flatten(1)
            if self.percentile == 0.:
                xmin = x.min(dim=1)[0] if not self.symmetric else torch.zeros(x.shape[0]).to(x.device)
                xmax = x.max(dim=1)[0] if not self.symmetric else x.abs().max(dim=1)[0]
            elif not self.symmetric:
                xmin = x.kthvalue(int(x.shape[1] * self.percentile) + 1, dim=1)[0]
                xmax = x.kthvalue(int(x.shape[1] * (1 + self.percentile)), dim=1)[0]
            else:
                xmin = torch.zeros(x.shape[0]).to(x.device)
                xmax = x.abs().kthvalue(int(x.shape[1] * (1 - self.percentile)), dim=1)[0]

        else:
            raise NotImplementedError(f'Granularity {self.granularity} not implemented.')
        
        xmin, xmax = self.update(xmin, xmax)

        return xmin, xmax

    def quantize(self, xmin: Tensor, xmax: Tensor):
        """ quantize according to the range
        Args:
            xmin (torch.Tensor): minimum value
            xmax (torch.Tensor): maximum value
        Returns:
            scale: quantization scale
            zero: quantization zero point
            qmin: quantization minimum value
            qmax: quantization maximum value
        """
        n_bits = self.n_bits

        if self.symmetric:
            if self.signed:
                qmax = (1 << (n_bits - 1)) - 1
                qmin = -(1 << (n_bits - 1))
                quant_range = float(qmax - qmin - 1) / 2
            else:
                qmax = (1 << n_bits) - 1
                qmin = 0
                quant_range = float(qmax - qmin)
            
            value_range = torch.max(xmin.abs(), xmax.abs())
            scale = value_range / quant_range
            zero = torch.zeros_like(scale)
        else:
            qmax = (1 << n_bits) - 1
            qmin = 0
            quant_range = float(qmax - qmin)

            value_range = xmax - xmin
            scale = value_range / quant_range
            zero = xmin / scale

        return scale, zero, qmin, qmax

    def __call__(self, x: Tensor, flag: str):
        """ estimate quantization range
        Args:
            x (torch.Tensor): input tensor, 
                shape (C, ...) for weight
                shape (N, C, ...) for activation
            flag (str): flag for weight or activation
        """
        xmin, xmax = self.range(x, flag)
        
        return self.quantize(xmin, xmax)


class MAMinMax(MinMax):
    """ Moving average min-max range estimator.
        During calibration, the minimum and maximum values of all input batches are recorded
        by moving average method with momentum.
        The quantization range is determined by the minimum and maximum values.

    Args:
        n_bits: number of bits
        symmetric: whether to use symmetric quantization
        signed: whether to use signed quantization
        granularity: quantization granularity, channel or tensor
        momentum: momentum of moving average, Default: 0.1

    Returns:
        scale: quantization scale
        zero: quantization zero point
        qmin: quantization minimum value
        qmax: quantization maximum value
    
    """
    def __init__(self, momentum: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

    def update(self, xmin, xmax):
        """ update min-max range by moving average method.
        Args:
            xmin (torch.Tensor): minimum value
            xmax (torch.Tensor): maximum value
        Returns:
            torch.Tensor: updated minimum value
            torch.Tensor: updated maximum value
        """
        if 'xmin' not in self.__dict__:
            self.xmin = xmin
            self.xmax = xmax
        elif self.momentum >= 0. and self.momentum <= 1.:
            self.xmin = self.momentum * xmin + (1 - self.momentum) * self.xmin
            self.xmax = self.momentum * xmax + (1 - self.momentum) * self.xmax
        else:
            self.xmin = torch.min(self.xmin, xmin)
            self.xmax = torch.max(self.xmax, xmax)

        return self.xmin, self.xmax
