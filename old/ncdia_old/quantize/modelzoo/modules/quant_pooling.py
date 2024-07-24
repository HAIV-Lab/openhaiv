"""
Quantized torch.nn.pooling
version: 0.0.2
update: 2024-01-09
"""
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from typing import Optional
from torch.nn.common_types import _size_any_t, _size_any_opt_t

from .quantizer import Quantizer


class QuantMaxPool2d(nn.MaxPool2d):
    """
    Quantized MaxPool2d layer

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
            Useful for :class:`torch.nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        a_setting: activation quantization setting
            n_bits: number of bits
            symmetric: whether to use symmetric quantization
            signed: whether to use signed quantization
            granularity: quantization granularity, channel or tensor
            range: quantization range, min_max
        device: device type
        dtype: data type

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in} + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)`
          and :math:`W_{out} = floor((W_{in} + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)`

    Examples::
    
            >>> m = QuantMaxPool2d(3, stride=2)
            >>> input = torch.randn(20, 16, 50, 32)
            >>> output = m(input)
            >>> print(output.size())
            torch.Size([20, 16, 24, 15])

    """
    def __init__(
        self,
        kernel_size: _size_any_t,
        stride: Optional[_size_any_t] = None,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        a_setting: dict = {},
        **kwargs
    ):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
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

        return F.max_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.dilation, ceil_mode=self.ceil_mode,
                            return_indices=self.return_indices)

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
            return F.max_pool2d((x + a_zero).mul_(a_scale), self.kernel_size, self.stride,
                                self.padding, self.dilation, ceil_mode=self.ceil_mode,
                                return_indices=self.return_indices)


class QuantAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """
    Quantized AdaptiveAvgPool2d layer

    Args:
        output_size: the target output size of the image of the form H x W.
            Can be a tuple (H, W) or a single H for a square image H x H
        a_setting: activation quantization setting
            n_bits: number of bits
            symmetric: whether to use symmetric quantization
            signed: whether to use signed quantization
            granularity: quantization granularity, channel or tensor
            range: quantization range,
    
    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = output\_size[0]` and :math:`W_{out} = output\_size[1]`

    Examples::
        
            >>> m = QuantAdaptiveAvgPool2d((5,7))
            >>> input = torch.randn(1, 64, 8, 9)
            >>> output = m(input)
            >>> print(output.size())
            torch.Size([1, 64, 5, 7])

    """
    def __init__(
        self,
        output_size: _size_any_opt_t,
        a_setting: dict = {},
        **kwargs
    ):
        super().__init__(output_size)
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

        return F.adaptive_avg_pool2d(x, self.output_size)
    
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
            return F.adaptive_avg_pool2d(
                (x + a_zero).mul_(a_scale), self.output_size)
