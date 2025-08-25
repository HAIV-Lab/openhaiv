import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncdia.models.net.der_net import SimpleLinear

from ncdia.utils import MODELS, Configs


@MODELS.register
class AdaptiveNet(nn.Module):
    """AdaptiveNet for incremental learning.

    Args:
        network (Configs): Network configuration.

    """

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
            self.args["type"] = "memo_resnet18"

        self.TaskAgnosticExtractor, _ = MODELS.build(self.args)
        self.TaskAgnosticExtractor.train()

        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []


        self.AdaptiveExtractors = nn.ModuleList()


    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.AdaptiveExtractors)

    @property
    def num_features(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.AdaptiveExtractors)

    def extract_vector(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [
            extractor(base_feature_map) for extractor in self.AdaptiveExtractors
        ]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):

        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [
            extractor(base_feature_map) for extractor in self.AdaptiveExtractors
        ]
        features = torch.cat(features, 1)
        logits = self.fc(features)  # {logits: self.fc(features)}
        out = {"logits": logits}

        aux_logits = self.aux_fc(features[:, -self.out_dim :])

        out.update({"aux_logits": aux_logits, "features": features})
        out.update({"base_features": base_feature_map})
        return out

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def update_fc(self, nb_classes):
        if "type" not in self.args:
            self.args["type"] = "memo_resnet18"
        _, _new_extractor = MODELS.build(self.args)
        _new_extractor = _new_extractor.cuda()
        if len(self.AdaptiveExtractors) == 0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(
                self.AdaptiveExtractors[-2].state_dict()
            )

        if self.out_dim is None:
            self.out_dim = 512
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

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma
