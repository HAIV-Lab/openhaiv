import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
class DERNET(nn.Module):
    """DERNET for incremental learning.

    Args:
        network (Configs): Network configuration.

    """

    cur_class = [9]

    def __init__(
        self,
        network: Configs,
        base_classes,
        num_classes,
        net_alice,
        mode="ft_cos",
    ) -> None:
        super().__init__()
        self.args = network.cfg
        self.args["pretrained"] = True
        self.args["num_classes"] = 1000
        if "type" not in self.args:
            self.args["type"] = "resnet18"

        self.out_dim = None
        self.fc = None
        self.aux_fc = None

        print("cur_class: ", self.cur_class)

        self.task_sizes = []
        self.convnets = nn.ModuleList()


    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    @property
    def num_features(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = []
        for convnet in self.convnets:
            convnet(x)
            features.append(convnet.out_features)
        features = torch.cat(features, 1)
        return features

    def forward(self, x):

        features = []
        for convnet in self.convnets:
            convnet(x)
            features.append(convnet.out_features)
            # print(convnet.out_features.shape)
        # features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)

        logits = self.fc(features)  # {logics: self.fc(features)}
        out = {"logits": logits}

        aux_logits = self.aux_fc(features[:, -self.out_dim :])

        out.update({"aux_logits": aux_logits, "features": features})
        return out

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(MODELS.build(self.args))
        else:
            if "type" not in self.args:
                self.args["type"] = "resnet18"
            self.convnets.append(MODELS.build(self.args).cuda())
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = 2048
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc.cuda()

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1).cuda()

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma
