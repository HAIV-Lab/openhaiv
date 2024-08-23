"""
Quantized torch.nn.Linear
version: 0.1.3
update: 2024-01-18
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .quantizer import Quantizer
from .range import BiasCorrect
from ncdia.quantize.engine import tpack, tunpack


class QuantLinear(nn.Linear):
    """
    Quantized Linear layer

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
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
        bias_correct: whether to correct bias with calibration data
            momentum: momentum of expected mean, Default: 0.1
        device: device type
        dtype: data type

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(out\_features, in\_features)`
        bias:   the learnable bias of the module of shape :math:`(out\_features)`

    Examples::

            >>> m = QuantLinear(20, 30)
            >>> input = torch.randn(128, 20)
            >>> output = m(input)
            >>> print(output.size())
            torch.Size([128, 30])

    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        w_setting: dict = {},
        a_setting: dict = {},
        bias_correct: dict = {},
        device=None,
        dtype=None,
        **kwargs
    ):
        bias = kwargs['_parameters']['bias'] is not None
        super().__init__(in_features, out_features, bias, device, dtype)

        self.weight.data = kwargs['_parameters']['weight']
        if bias:
            self.bias.data = kwargs['_parameters']['bias']
        elif bias_correct:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
            self.bias.data = self.bias.data.to(self.weight.device)

        self.w_quantizer = Quantizer(**w_setting, flag='weight', n_channels=out_features, dim=self.weight.dim())
        self.a_quantizer = Quantizer(**a_setting, flag='activation', n_channels=in_features, dim=2)
        self.bias_correct = bias_correct

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
            bias = self.corrector(x, self.weight - weight,
                                  func=F.linear)
            bias = self.bias + bias
            return F.linear(x, weight, bias)
        
        return F.linear(x, weight, self.bias)
    
    def pack(self, calibrated=True):
        """ Pack the quantized layer.
            Weights are quantized and add scale and zero_point.
        Args:
            calibrated (bool): the layer is calibrated or not
        """
        self.requires_grad_(False)
        self.a_quantizer.pack(None)

        if getattr(self, 'corrector', None) and calibrated:
            weight = self.w_quantizer(self.weight)
            bias = self.corrector(None, self.weight - weight,
                                  func=F.linear)
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
            return F.linear((x + a_zero).mul_(a_scale),
                            (self.weight + self.w_zero).mul_(self.w_scale),
                            self.bias)
    
    def weight_pack(self):
        self.weight.data, self.w_des = tpack(self.weight, self.w_des[0],
                                             True if self.w_des[1] else False)

    def extra_repr(self):
        return super().extra_repr() + \
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
