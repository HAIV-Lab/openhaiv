import copy
import logging
import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ncdia.utils import MODELS, Configs


@MODELS.register
class SimpleLinear(nn.Module):
    """
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    """

    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="linear")
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


@MODELS.register
class ResNet_Base(nn.Module):
    """Normal Resnet

    Args:
        network (Configs): Network configuration.

    """

    def __init__(
        self,
        network: Configs,
        checkpoint: str = None,
        loss: str = "CrossEntropyLoss",  #
        **kwargs
    ) -> None:
        super().__init__()
        self.args = network.cfg
        self.args["pretrained"] = True
        num_classes_true = self.args["num_classes"]
        self.args["num_classes"] = 1000

        self.network = MODELS.build(copy.deepcopy(self.args))
        num_features = self.network.fc.in_features  # 获取输入特征维度
        self.network.fc = torch.nn.Linear(
            num_features, num_classes_true
        )  # 替换为新的全连接层
        self.out_features = None
        self.loss = loss
        if checkpoint:
            print("load_checkpoint")
            state_dict = torch.load(checkpoint)
            state_dict = {
                k.replace("network.", ""): v
                for k, v in state_dict["state_dict"].items()
            }
            self.network.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        feature1 = self.network.relu(self.network.bn1(self.network.conv1(x)))
        feature1 = self.network.maxpool(feature1)
        feature2 = self.network.layer1(feature1)
        feature3 = self.network.layer2(feature2)
        feature4 = self.network.layer3(feature3)
        feature5 = self.network.layer4(feature4)
        feature5 = self.network.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = self.network.fc(feature)

        # x = self.network(x)
        return logits_cls

    def forward_feature(self, x):
        feature1 = self.network.relu(self.network.bn1(self.network.conv1(x)))
        feature1 = self.network.maxpool(feature1)
        feature2 = self.network.layer1(feature1)
        feature3 = self.network.layer2(feature2)
        feature4 = self.network.layer3(feature3)
        feature5 = self.network.layer4(feature4)
        feature5 = self.network.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = self.network.fc(feature)

        # x = self.network(x)
        return logits_cls, feature

    def get_features(self, x):
        feature1 = self.network.relu(self.network.bn1(self.network.conv1(x)))
        feature1 = self.network.maxpool(feature1)
        feature2 = self.network.layer1(feature1)
        feature3 = self.network.layer2(feature2)
        feature4 = self.network.layer3(feature3)
        feature5 = self.network.layer4(feature4)
        feature5 = self.network.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = self.network.fc(feature)

        # x = self.network(x)
        return feature
