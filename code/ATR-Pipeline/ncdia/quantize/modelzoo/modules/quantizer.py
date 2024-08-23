"""
Quantizer
version: 0.1.7
update: 2024-01-17
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
from copy import deepcopy

from .range import RANGES


class Round(Function):
    """
    Round function:
    Compute x = round(x/scale - zero) and backpropagate gradients

    Args:
        x (Tensor): input tensor
        scale (Tensor): scale tensor
        zero (Tensor): zero point tensor

    Returns:
        rounded tensor
    """
    @staticmethod
    def forward(ctx, x, scale, zero):
        ctx.save_for_backward(x, scale, zero)
        return (x/scale - zero).round()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale, zero = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_x = grad_input / scale
        grad_scale = -grad_input * x / scale ** 2
        grad_zero = -grad_input
        return grad_x, grad_scale, grad_zero


class Quantizer(nn.Module):
    """
    Quantizer

    Args:
        n_bits (int): number of bits
        symmetric (bool): whether to use symmetric quantization
        signed (bool): whether to use signed quantization
        granularity (str): quantization granularity, channel or layer
        range (dict): quantization range estimator
            name (str): range estimator name
        adaound (dict): adaround setting
        flag (str): quantization flag, weight or activation
        n_channels (int): number of channels, default 1
        dim (int): dimension of input tensor, default 4
        static_scale (tensor): a fixed scale factor to multiple with the calibrated scale, default None

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *)`, same shape as the input, simulated 
          when self.packed is False
        - Output: :math:`(N, *)`, :math:`(N, *)`, :math:`(N, *)`,
          output tensor, scale and zero point, same shape as the input,
          packed when self.packed is True

    Examples::
    
            >>> m = Quantizer()
            >>> input = torch.randn(10)
            >>> output = m(input)
            >>> print(input)
            tensor([ 0.1427, -0.0800,  0.4543, -0.3333, -0.2124, -0.3480, -0.1095,  0.0853,
                    -0.0261, -0.0573])
            >>> print(output)
            tensor([ 0.1426, -0.0801,  0.4541, -0.3333, -0.2124, -0.3481, -0.1094,  0.0853,
                    -0.0261, -0.0573])
    
    """
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = True,
        signed: bool = True,
        granularity: str = 'layer',
        range: dict = {'name': 'maminmax'},
        adaround: dict = {},
        flag: str = 'weight',
        n_channels: int = 1,
        dim: int = 4,
        static_scale: Tensor = None,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.signed = signed
        self.granularity = granularity
        self.flag = flag
        self.n_channels = n_channels
        self.dim = dim

        range = deepcopy(range)
        self.range = range.pop('name')
        self.range_estimator = RANGES[self.range](
            n_bits=self.n_bits,
            symmetric=self.symmetric,
            signed=self.signed,
            granularity=self.granularity,
            **range)
        
        self.adaround = adaround
        if adaround:
            self.adaround = RANGES['adaround'](**adaround)
        
        if granularity in ['L', 'Layer', 'layer']:
            self.n_channels = 1

        self.scale = nn.Parameter(torch.ones(self.n_channels).view(*self.shape))
        self.zero = nn.Parameter(torch.zeros(self.n_channels).view(*self.shape))
        self.register_buffer('qmin', torch.tensor(-(2**(self.n_bits-1))))
        self.register_buffer('qmax', torch.tensor(2**(self.n_bits-1)-1))

        self._static_scale = None
        if static_scale is not None:
            self._static_scale = nn.Parameter(static_scale.view(*self.shape))

        self.quantized = False
        self.packed = False
    
    @property
    def shape(self):
        """ the shape used for scale and zero to be reshaped
        """
        shape = [1] * self.dim
        if self.flag == 'activation': shape[1] = -1
        else: shape[0] = -1
        return shape
    
    @property
    def static_scale(self):
        """ static scale factor to multiple with the calibrated scale
        """
        if self._static_scale is None:
            return torch.ones_like(self.scale)
        return self._static_scale
        
    def quant(self, quantized: bool = True):
        """ shift between simulated (quantized=True) and original (quantized=False) mode
        """
        self.quantized = quantized or False

    def set_state(self, state: dict):
        """ set static variables for quantizer
        Args:
            state (dict): static variables, including "static_scale"
        """
        device = self.scale.device
        static_scale = state['static_scale'].view_as(self.scale).to(device)
        if self._static_scale:
            self._static_scale.data = static_scale
        else:
            self._static_scale = nn.Parameter(static_scale)
        # TODO: convert to channel-wise mode according to static_scale
        assert len(self._static_scale) == self.n_channels, \
            'static_scale has different number of channels with quantizer'

    def round(self, x: Tensor, scale: Tensor, zero: Tensor) -> Tensor:
        """ round tensor
        Args:
            x: input tensor
            scale: scale tensor
            zero: zero point tensor
        Returns:
            rounded tensor
        """
        if self.adaround:
            return self.adaround(x/scale - zero)
        return Round.apply(x, scale, zero)  # (x/scale - zero).round()

    def simulate(self, x: Tensor) -> Tensor:
        """ simulate quantization
        Args:
            x: input tensor,
                shape (C, ...) for weight
                shape (N, C, ...) for activation
        Returns:
            quantized tensor
        """
        scale, zero = self.scale, self.zero
        static_scale = self.static_scale

        x = self.round(x, scale, zero).clamp(self.qmin, self.qmax)  # quantize

        if not self.packed:
            x = (x + zero).mul(scale * static_scale)  # dequantize
            return x
        else:
            return x, scale * static_scale, zero
    
    def pack(self, x: Tensor) -> Tensor:
        """ pack weight and return quantized tensor
        Args:
            x: input tensor,
                shape (C, ...) for weight
        Returns:
            quantized tensor
        """
        self.packed = True
        if self.flag == 'weight':
            scale, zero = self.scale, self.zero
            static_scale = self.static_scale
            x = self.round(x, scale, zero).clamp(self.qmin, self.qmax)
            return x, scale * static_scale, zero
        return x, None, None
    
    def calibrate(self, x: Tensor):
        """ calibrate quantizer
        Args:
            x: input tensor,
                shape (C, ...) for weight
                shape (N, C, ...) for activation
        """
        device = self.scale.device
        scale, zero, qmin, qmax = \
            self.range_estimator(x, self.flag)

        self.scale.data = scale.view(*self.shape).to(device)
        self.zero.data = zero.view(*self.shape).to(device)
        self.qmin.copy_(torch.tensor(qmin).to(device))
        self.qmax.copy_(torch.tensor(qmax).to(device))
    
    def forward(self, x: Tensor) -> Tensor:
        """ forward pass
        Args:
            x: input tensor
        Returns:
            quantized and simulated tensor, if self.quantized
            input tensor, otherwise
        """
        if not self.quantized or self.n_bits >= 32:
            if not self.packed:
                return x * self.static_scale
            else:
                return x, self.static_scale, torch.zeros_like(x)
            
        return self.simulate(x)
    
    def extra_repr(self):
        return f'n_bits={self.n_bits}' + \
               f', symmetric={self.symmetric}' + \
               f', signed={self.signed}' + \
               f', granularity={self.granularity}' + \
               f', range={self.range}' + \
               f', n_channels={self.n_channels}'
    
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs
    ):
        device = self.scale.device
        self.scale.data = state_dict[prefix+'scale'].to(device)
        self.zero.data = state_dict[prefix+'zero'].to(device)
        if prefix+'static_scale' in state_dict:
            self.set_state({'static_scale': state_dict[prefix+'static_scale']})
        
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs)
