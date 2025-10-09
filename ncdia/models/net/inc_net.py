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
class BaseNet(nn.Module):
    """BaseNet for incremental learning.

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
        self.args["num_classes"] = num_classes
        if "type" not in network:
            self.args["type"] = "resnet50"
        self.convnet = MODELS.build(self.args)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        self.convnet(x)
        return self.convnet.out_features

    def forward(self, x):
        x = self.convnet(x)
        features = self.convnet.out_features
        out = self.fc(features)
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        # out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


@MODELS.register
class IncrementalNet(BaseNet):
    """BaseNet for incremental learning.

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
        super().__init__(
            network, base_classes, num_classes, net_alice, mode
        )
        self.update_fc(num_classes)

    def update_fc(self, nb_classes):
        # use the actual feature dimension from the backbone (resnet18 -> 512, resnet50 -> 2048)
        # Detect feature dimension from the backbone in a robust way.
        # Try several common attributes used by different backbones:
        #  - custom nets may expose `out_dim`
        #  - torchvision ResNet exposes `fc.in_features`
        #  - mobilenet-like nets expose `classifier[-1].in_features`
        if not hasattr(self, "convnet") or self.convnet is None:
            raise RuntimeError("convnet is not initialized before update_fc")

        in_dim = None
        # custom attribute used elsewhere in the code
        if hasattr(self.convnet, "out_dim"):
            in_dim = getattr(self.convnet, "out_dim")
        # torchvision ResNet / similar
        elif hasattr(self.convnet, "fc") and hasattr(self.convnet.fc, "in_features"):
            in_dim = int(getattr(self.convnet.fc, "in_features"))
        # mobilenet style classifier (list/nn.Sequential)
        elif hasattr(self.convnet, "classifier"):
            try:
                cls = getattr(self.convnet, "classifier")
                # classifier might be Sequential; take last module's in_features
                if isinstance(cls, (list, tuple)):
                    last = cls[-1]
                else:
                    # nn.Sequential or Module
                    last = list(cls.children())[-1]
                if hasattr(last, "in_features"):
                    in_dim = int(getattr(last, "in_features"))
            except Exception:
                in_dim = None

        if in_dim is None:
            raise RuntimeError(
                "Unable to infer feature dimension from convnet. "
                "Please ensure the backbone exposes `out_dim`, or `fc.in_features`, "
                "or `classifier[-1].in_features`."
            )

        fc = self.generate_fc(in_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        x = self.convnet(x)
        features = self.convnet.out_features
        out = self.fc(features)
        # out.update(x)

        return out
