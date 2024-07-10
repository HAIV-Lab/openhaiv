"""
Quantized torch.nn.Conv2d
version: 0.1.4
update: 2024-01-18
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

from torch.nn.common_types import _size_2_t
from typing import Union

from .quantizer import Quantizer
from .range import BiasCorrect
from ncdia_old.quantize.engine import tpack, tunpack


class QuantConv2d(nn.Conv2d):
    """ Quantized 2D convolution layer

    Args:
        in_channels: number of channels in the input image
        out_channels: number of channels produced by the convolution
        kernel_size: size of the convolving kernel
        stride: stride of the convolution. Default: 1
        padding: zero-padding added to both sides of the input. Default: 0
        dilation: spacing between kernel elements. Default: 1
        groups: number of blocked connections from input channels to output channels. Default: 1
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        padding_mode: ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'zeros'``
        w_setting: weight quantization setting
            n_bits: number of bits
            symmetric: whether to use symmetric quantization
            signed: whether to use signed quantization
            granularity: quantization granularity, channel or tensor
            range: quantization range, min_max
        a_setting: activation quantization setting
            n_bits: number of bits
            symmetric: whether to use symmetric quantization
            signed: whether to use signed quantization
            granularity: quantization granularity, channel or tensor
            range: quantization range, min_max
        bn_folding: batch normalization folding setting
            running_mean: running mean
            running_var: running variance
            weight: batch normalization weight
            bias: batch normalization bias
            eps: batch normalization eps
        bias_correct: whether to correct bias with calibration data
            momentum: momentum of expected mean, Default: 0.1
        device: device type
        dtype: data type

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          :math:`H_{out} = \\lfloor \\frac{H_{in} + 2 \\times \\text{padding}[0] - \\text{dilation}[0]
          \\times (\\text{kernel_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\rfloor`
          and :math:`W_{out} = \\lfloor \\frac{W_{in} + 2 \\times \\text{padding}[1] - \\text{dilation}[1]
          \\times (\\text{kernel_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\rfloor`

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\\text{out_channels}, \\frac{\\text{in_channels}}{\\text{groups}},`
            :math:`\\text{kernel_size[0]}, \\text{kernel_size[1]})`
        bias:   the learnable bias of the module of shape :math:`(\\text{out_channels})`

    Examples::
    
            >>> m = QuantConv2d(16, 33, 3, stride=2)
            >>> input = torch.randn(20, 16, 50, 100)
            >>> output = m(input)
            >>> print(output.size())
            torch.Size([20, 33, 24, 49])
    
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        padding_mode: str = 'zeros',
        w_setting: dict = {},
        a_setting: dict = {},
        bn_folding: dict = {},
        bias_correct: dict = {},
        device=None,
        dtype=None,
        **kwargs
    ):
        bias = kwargs['_parameters']['bias'] is not None
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, device, dtype)

        self.weight.data = kwargs['_parameters']['weight']
        if bias:
            self.bias.data = kwargs['_parameters']['bias']
        elif bn_folding or bias_correct:
            self.bias = Parameter(torch.zeros(out_channels, device=device, dtype=dtype))
            self.bias.data = self.bias.data.to(self.weight.device)

        self.w_quantizer = Quantizer(**w_setting, flag='weight', n_channels=out_channels, dim=self.weight.dim())
        self.a_quantizer = Quantizer(**a_setting, flag='activation', n_channels=in_channels, dim=4)
        self.bn_folding = bn_folding
        self.bias_correct = bias_correct

        if bn_folding:
            self.bias.data = (
                (bn_folding['bias']
                + (self.bias.data - bn_folding['running_mean'])
                * bn_folding['weight']
                / torch.sqrt(bn_folding['running_var'] + bn_folding['eps']))\
                    .reshape(-1))

            if not bn_folding.get('into_scale', False):
                self.weight.data = (
                    self.weight.data
                    * (bn_folding['weight']
                    / torch.sqrt(bn_folding['running_var'] + bn_folding['eps']))\
                        .reshape(-1, 1, 1, 1))
            else:
                self.w_quantizer.set_state({'static_scale': (
                    (bn_folding['weight']
                    / torch.sqrt(bn_folding['running_var'] + bn_folding['eps']))\
                        .reshape(-1))})
        
        if bias_correct:
            self.corrector = BiasCorrect(**bias_correct)

        self.calibrating = False
        self.packed = False
    
    def calibrate(self, x: Tensor):
        """ calibrate weight and activation quantizer.
        """
        self.a_quantizer.calibrate(x.detach().clone())
        self.w_quantizer.calibrate(self.weight.detach().clone())

        if getattr(self, 'corrector', None):
            self.corrector.calibrate(x.detach().clone())

    def _forward(self, x: Tensor) -> Tensor:
        if self.calibrating:
            self.calibrate(x)

        x = self.a_quantizer(x)
        weight = self.w_quantizer(self.weight)

        if getattr(self, 'corrector', None):
            ori_weight = self.weight * self.w_quantizer.static_scale
            bias = self.corrector(x, ori_weight - weight,
                                  func=self._conv_forward)
            if bias.dim() == 0:
                bias = bias.extend(self.bias.shape)
            else:
                bias = bias.mean(dim=(1, 2))
            bias = self.bias + bias
            return self._conv_forward(x, weight, bias)
        
        return self._conv_forward(x, weight, self.bias)
    
    def pack(self, calibrated=True):
        """ Pack the quantized layer.
            Weights are quantized and add scale and zero_point.
        Args:
            calibrated (bool): the layer is calibrated or not
        """
        self.requires_grad_(False)
        self.a_quantizer.pack(None)

        if getattr(self, 'corrector', None) and calibrated:
            ori_weight = self.weight * self.w_quantizer.static_scale
            weight = self.w_quantizer(self.weight)
            bias = self.corrector(None, ori_weight - weight,
                                  func=self._conv_forward)
            if bias.dim() == 0:
                bias = bias.extend(self.bias.shape)
            else:
                bias = bias.mean(dim=(1, 2))
            self.bias.data += bias

        if self.w_quantizer:
            weight, w_scale, w_zero = self.w_quantizer.pack(self.weight)
            self.register_buffer('w_scale', w_scale)
            self.register_buffer('w_zero', w_zero)

            self.weight.data, w_des = tpack(weight, self.w_quantizer.n_bits, 
                                            self.w_quantizer.signed)
            self.register_buffer('w_des', w_des)
        
        self.w_quantizer = None
        self.corrector = None
        self.packed = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.packed:
            return self._forward(x)
        else:
            x, a_scale, a_zero = self.a_quantizer(x)
            return self._conv_forward((x + a_zero).mul_(a_scale),
                                      (self.weight + self.w_zero).mul_(self.w_scale),
                                      self.bias)
        
    def weight_pack(self):
        self.weight.data, self.w_des = tpack(self.weight, self.w_des[0], 
                                             True if self.w_des[1] else False)

    def extra_repr(self):
        return super().extra_repr() + \
            (f', bn_folding=True' if self.bn_folding else '') + \
            (f', bf_into_scale=True' if self.bn_folding.get('into_scale', False) else '') + \
            (f', bias_correct=True' if self.bias_correct else '')
    
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, 
        strict, missing_keys, unexpected_keys, error_msgs
    ):
        device = self.weight.device
        if prefix + 'w_des' in state_dict:
            self.pack(calibrated=False)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, 
            strict, missing_keys, unexpected_keys, error_msgs)
        
        if prefix + 'w_des' in state_dict:
            self.weight.data = tunpack(self.weight, self.w_des)
        self.to(device)
