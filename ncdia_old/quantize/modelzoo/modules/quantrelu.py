"""
Quantized torch.nn.ReLU
version: 0.0.3
update: 2024-01-09
"""
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .quantizer import Quantizer


class QuantReLU(nn.ReLU):
    """
    Quantized ReLU layer

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
        a_setting: activation quantization setting
            n_bits: number of bits
            symmetric: whether to use symmetric quantization
            signed: whether to use signed quantization
            granularity: quantization granularity, channel or tensor
            range: quantization range, min_max
        device: device type
        dtype: data type

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *)`, same shape as the input.

    Examples::
    
            >>> m = QuantReLU()
            >>> input = torch.randn(2)
            >>> output = m(input)
            >>> print(output)
            tensor([0.0000, 0.2763])

            >>> m = QuantReLU(inplace=True)
            >>> input = torch.randn(2)
            >>> output = m(input)
            >>> print(output)
            tensor([0.0000, 0.2763])
            >>> print(input)
            tensor([0.0000, 0.2763])

    """
    def __init__(
        self,
        inplace: bool = False,
        a_setting: dict = {},
        **kwargs
    ):
        super().__init__(inplace)
        self.a_quantizer = Quantizer(**a_setting, flag='activation')
        self.calibrating = False
        self.packed = False

    def calibrate(self, x: Tensor):
        """ calibrate activation quantizer.
        """
        self.a_quantizer.calibrate(x.detach().clone())

    def _forward(self, x: Tensor) -> Tensor:
        if self.calibrating:
            self.calibrate(x)

        x = self.a_quantizer(x)

        return F.relu(x, inplace=self.inplace)
    
    def pack(self):
        """ Pack the quantized layer.
        """
        self.requires_grad_(False)
        self.a_quantizer.pack(None)
        self.packed = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.packed:
            return self._forward(x)
        else:
            x, a_scale, a_zero = self.a_quantizer(x)
            return F.relu((x + a_zero).mul_(a_scale), inplace=self.inplace)
