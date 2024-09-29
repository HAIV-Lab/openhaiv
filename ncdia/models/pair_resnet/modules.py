import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torchvision.utils import _log_api_usage_once
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models._api import WeightsEnum

from typing import Any, Callable, List, Optional, Type, Union, Tuple

__all__ = ['BasicBlock', 'Bottleneck', 'ResNet', '_pair_resnet']


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# The HybridFusion module
class HybridFusion(nn.Module):
    def __init__(self, planes: int) -> None:
        super(HybridFusion, self).__init__()

        self.CBR1_concat = nn.Sequential(nn.Conv2d(2*planes, planes, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(planes),
                                         nn.ReLU(inplace=True))
        self.CBR1_max = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(planes),
                                      nn.ReLU(inplace=True))
        self.CBR1_mul = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(planes),
                                      nn.ReLU(inplace=True))
        
        self.CBR2_concat = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(planes),
                                         nn.ReLU(inplace=True))
        self.CBR2_max = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(planes),
                                      nn.ReLU(inplace=True))
        self.CBR2_mul = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(planes),
                                      nn.ReLU(inplace=True))
        
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x: Tensor, y:Tensor) -> Tensor:
        # The Concat Branch
        F_concat = self.CBR2_concat(self.CBR1_concat(torch.concat((x, y), axis=1)))
        
        # The Add Branch
        F_max = self.CBR2_max(self.CBR1_max(torch.max(x, y)))
        
        # The mul Branch
        F_mul = self.CBR2_mul(self.CBR1_mul(torch.mul(x, y)))

        result = F_concat + self.alpha*F_max + self.beta*F_mul

        return result 


# 普通残差模块
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Bottleneck残差模块
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as resnet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

# 模型ResNet具体实现类
class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.features = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        self.features = x[:]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# SoftMax fusion Paired ResNet
class PairSMResNet(nn.Module):
    def __init__(
        self, 
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        fusion_type: str, 
        **kwargs
    ) -> None:
        super().__init__()
        self.model_a = ResNet(block, layers, **kwargs)
        self.model_b = ResNet(block, layers, **kwargs)
        self.fusion_type = fusion_type
        self.features = None

    def forward(self, input: Tuple[Tensor, Tensor]) -> Tensor:
        x, y = input[0], input[1]
        x = self.model_a(x)
        y = self.model_b(y)
        
        if self.fusion_type == 'multi':
            return torch.mul(x, y)
        elif self.fusion_type == 'add':
            return torch.add(x, y)
        elif self.fusion_type == 'max':
            return torch.max(x, y)
        else:
            raise ValueError(f"Please set a true fusion_type value for SoftMax fusion!!!.")


# Feature extract by n blocks
class FeatureExtractor(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            fusion_pos: str, 
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if 'block' in fusion_pos:
            self.fusion_at_block = int(fusion_pos[-1])
        elif fusion_pos == 'extend':
            self.fusion_at_block = 5
        elif fusion_pos == 'MultiScale':
            self.fusion_at_block = 5
        else:
            raise ValueError(f"Please set a true fusion_pos value!!!.")

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if self.fusion_at_block > 1:
            self.layer1 = self._make_layer(block, 64, layers[0])
        if self.fusion_at_block > 2:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0])
        if self.fusion_at_block > 3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
        if self.fusion_at_block > 4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2])

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        if self.fusion_at_block > 1:
            x = self.layer1(x)
        if self.fusion_at_block > 2:
            x = self.layer2(x)   
        if self.fusion_at_block > 3:
            x = self.layer3(x)
        if self.fusion_at_block > 4:
            x = self.layer4(x)

        return x


# The Convolution operation on fusion features (raw width as raw ResNet)
class FeatureRecognition(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        fusion_pos: str, 
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.fusion_pos = fusion_pos

        if 'block' in self.fusion_pos:
            self.fusion_at_block = int(fusion_pos[-1])
        elif self.fusion_pos == 'extend':
            self.fusion_at_block = 5
        elif self.fusion_pos == 'MultiScale':
            self.fusion_at_block = 1
        else:
            raise ValueError(f"Please set a true fusion_pos value!!!.")

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        channel_nums = [64, 64, 128, 256, 512]
        self.inplanes = channel_nums[0] if self.fusion_at_block == 1 \
                            else channel_nums[self.fusion_at_block - 1]*block.expansion

        if self.fusion_at_block <= 1:
            self.layer1 = self._make_layer(block, 64, layers[0])
        if self.fusion_at_block <= 2:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                            dilate=replace_stride_with_dilation[0])
        if self.fusion_at_block <= 3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
        if self.fusion_at_block <= 4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2])
        if self.fusion_pos == 'extend':
            self.layer5 = self._make_layer(block, 512, 2)
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes,bias=False)
        self.features = None

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.fusion_at_block <= 1:
            x = self.layer1(x)
        if self.fusion_at_block <= 2:
            x = self.layer2(x)
        if self.fusion_at_block <= 3:
            x = self.layer3(x)
        if self.fusion_at_block <= 4:
            x = self.layer4(x)

        if self.fusion_pos == 'extend':
            x = self.layer5(x)

        self.features = x[:]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# The fusion block
class FeatureFusionModel(nn.Module):
    def __init__(
        self, 
        fusion_pos: str, 
        fusion_type: str, 
        expansion: int = 1
    ) -> None:
        super(FeatureFusionModel, self).__init__()
        self.fusion_type = fusion_type
        
        if 'block' in fusion_pos:
            fusion_at_block = int(fusion_pos[-1])
        elif fusion_pos == 'extend':
            fusion_at_block = 5
        else:
            raise ValueError(f"Please set a true fusion_pos value!!!.")

        channel_nums = [64, 64, 128, 256, 512]
        planes = channel_nums[0] if fusion_at_block == 1 else channel_nums[fusion_at_block - 1]*expansion
        inplace = 2*planes if self.fusion_type == 'concat' else planes
        
        if self.fusion_type == 'hybrid':
            self.HFModule = HybridFusion(planes)
        else:
            self.CBR1 = nn.Sequential(nn.Conv2d(inplace, planes, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(planes),
                                      nn.ReLU(inplace=True))

            self.CBR2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(planes),
                                      nn.ReLU(inplace=True))
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self.fusion_type == 'concat':
            feature = self.CBR2(self.CBR1(torch.concat((x, y), axis=1)))
        elif self.fusion_type == 'max':
            feature = self.CBR2(self.CBR1(torch.max(x, y)))
        elif self.fusion_type == 'add':
            feature = self.CBR2(self.CBR1(torch.add(x, y)))
        elif self.fusion_type == 'mul':
            feature = self.CBR2(self.CBR1(torch.mul(x, y)))
        elif self.fusion_type == 'hybrid':
            feature = self.HFModule(x, y)

        return feature


# Pair ResNet using max, multi, add and conv1x1 to get fusion feature
class PairOptResNet(nn.Module):
    def __init__(
        self, 
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        fusion_pos: str, 
        fusion_type: str,
        **kwargs
    ) -> None:
        super().__init__()
        self.feature_extract_a = FeatureExtractor(block, layers, fusion_pos, **kwargs)
        self.feature_extract_b = FeatureExtractor(block, layers, fusion_pos, **kwargs)
        self.fusion_model = FeatureFusionModel(fusion_pos, fusion_type, expansion=block.expansion)
        self.recognizer = FeatureRecognition(block, layers, fusion_pos, **kwargs)
        self.features = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input: Tuple[Tensor, Tensor]) -> Tensor:
        x, y = input[0], input[1]
        feature_a = self.feature_extract_a(x)
        feature_b = self.feature_extract_b(y)
        fusion_feature = self.fusion_model(feature_a, feature_b)
        result = self.recognizer(fusion_feature)

        self.features = self.recognizer.features
        return result


# The MultiScaleResNet module
class PairMultiScaleResNet(nn.Module):
    def __init__(
        self, 
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        fusion_pos: str, 
        fusion_type: str,
        **kwargs
    ) -> None:
        super().__init__()
        self.feature_extract_a = FeatureExtractor(block, layers, fusion_pos, **kwargs)
        self.feature_extract_b = FeatureExtractor(block, layers, fusion_pos, **kwargs)

        self.fusion_model1 = FeatureFusionModel('block1', fusion_type, expansion=block.expansion)
        self.fusion_model2 = FeatureFusionModel('block2', fusion_type, expansion=block.expansion)
        self.fusion_model3 = FeatureFusionModel('block3', fusion_type, expansion=block.expansion)
        self.fusion_model4 = FeatureFusionModel('block4', fusion_type, expansion=block.expansion)
        self.fusion_model5 = FeatureFusionModel('block5', fusion_type, expansion=block.expansion)
        
        self.combination2 = FeatureFusionModel('block2', fusion_type, expansion=block.expansion)
        self.combination3 = FeatureFusionModel('block3', fusion_type, expansion=block.expansion)
        self.combination4 = FeatureFusionModel('block4', fusion_type, expansion=block.expansion)
        self.combination5 = FeatureFusionModel('block5', fusion_type, expansion=block.expansion)

        self.recognizer = FeatureRecognition(block, layers, fusion_pos, **kwargs)
        self.features = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input: Tuple[Tensor, Tensor]) -> Tensor:
        # The multi-granularity features
        x, y = input[0], input[1]
        x = self.feature_extract_a.conv1(x)
        x = self.feature_extract_a.bn1(x)
        x = self.feature_extract_a.relu(x)
        x1 = self.feature_extract_a.maxpool(x)
        x2 = self.feature_extract_a.layer1(x1)
        x3 = self.feature_extract_a.layer2(x2)
        x4 = self.feature_extract_a.layer3(x3)
        x5 = self.feature_extract_a.layer4(x4)

        y = self.feature_extract_b.conv1(y)
        y = self.feature_extract_b.bn1(y)
        y = self.feature_extract_b.relu(y)
        y1 = self.feature_extract_b.maxpool(y)
        y2 = self.feature_extract_b.layer1(y1)
        y3 = self.feature_extract_b.layer2(y2)
        y4 = self.feature_extract_b.layer3(y3)
        y5 = self.feature_extract_b.layer4(y4)

        # The multi-granularity feature fusion
        fusion_feature1 = self.fusion_model1(x1, y1)
        fusion_feature2 = self.fusion_model2(x2, y2)
        fusion_feature3 = self.fusion_model3(x3, y3)
        fusion_feature4 = self.fusion_model4(x4, y4)
        fusion_feature5 = self.fusion_model5(x5, y5)

        # fusion 
        feature = self.combination2(fusion_feature2, self.recognizer.layer1(fusion_feature1))
        feature = self.combination3(fusion_feature3, self.recognizer.layer2(feature))
        feature = self.combination4(fusion_feature4, self.recognizer.layer3(feature))
        feature = self.combination5(fusion_feature5, self.recognizer.layer4(feature))

        self.features = feature[:]
        feature = self.recognizer.avgpool(feature)
        feature = torch.flatten(feature, 1)
        feature = self.recognizer.fc(feature)

        return feature


# 定义根据参数加载对应的resnet模型方法
def _pair_resnet(
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        weights: Optional[WeightsEnum],
        progress: bool,
        fusion_pos: str,
        fusion_type: str,
        **kwargs: Any,
    ):
    if fusion_pos == 'SoftMax':
        model = PairSMResNet(block, layers, fusion_type, **kwargs)
        if weights is not None:
            model_dict = model.state_dict()
            state_dict = weights.get_state_dict(progress=progress)

            pair_state_dict = {}
            for k, v in state_dict.items():
                if k not in ['fc.weight', 'fc.bias']:
                    pair_state_dict['model_a.' + k] = v
                    pair_state_dict['model_b.' + k] = v

            model_dict.update(pair_state_dict)
            model.load_state_dict(model_dict)

    else:
        if fusion_pos == 'MultiScale':
            model = PairMultiScaleResNet(block, layers, fusion_pos, fusion_type, **kwargs)
        else:
            model = PairOptResNet(block, layers, fusion_pos, fusion_type, **kwargs)

        if weights is not None:
            model_dict = model.state_dict()
            state_dict = weights.get_state_dict(progress=progress)

            pair_state_dict = {}
            for k, v in model_dict.items():
                if 'fc.weight' in k or 'fc.bias' in k:
                    continue

                if 'feature_extract_a' in k:
                    new_k = k.replace("feature_extract_a.", "")
                    if new_k in state_dict.keys():
                        pair_state_dict[k] = state_dict[new_k]

                if 'feature_extract_b' in k:
                    new_k = k.replace("feature_extract_b.", "")
                    if new_k in state_dict.keys():
                        pair_state_dict[k] = state_dict[new_k]
                
                if 'recognizer' in k:
                    new_k = k.replace("recognizer.", "")
                    if new_k in state_dict.keys():
                        pair_state_dict[k] = state_dict[new_k]

            model_dict.update(pair_state_dict)
            model.load_state_dict(model_dict)
    return model