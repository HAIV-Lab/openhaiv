import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncdia.utils import MODELS, Configs


@MODELS.register
class AliceNET(nn.Module):
    """
    AliceNET for incremental learning.

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

        self.mode = mode
        self.base_classes = base_classes
        self.num_classes = num_classes
        self.net_alice = net_alice

        # pretrained=True follow TOPIC, models for cub is imagenet pre-trained.
        # https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
        network = network.cfg
        network["pretrained"] = True
        network["num_classes"] = num_classes
        if "type" not in network:
            network["type"] = "resnet18"
        self.encoder = MODELS.build(network)
        # Detect feature dimension from the encoder/backbone in a robust way.
        num_features = None
        if hasattr(self.encoder, "out_dim"):
            num_features = getattr(self.encoder, "out_dim")
        elif hasattr(self.encoder, "fc") and hasattr(self.encoder.fc, "in_features"):
            num_features = int(getattr(self.encoder.fc, "in_features"))
        elif hasattr(self.encoder, "classifier"):
            try:
                cls = getattr(self.encoder, "classifier")
                if isinstance(cls, (list, tuple)):
                    last = cls[-1]
                else:
                    last = list(cls.children())[-1]
                if hasattr(last, "in_features"):
                    num_features = int(getattr(last, "in_features"))
            except Exception:
                num_features = None

        if num_features is None:
            # fallback to the original hardcoded value but warn
            try:
                import warnings

                warnings.warn(
                    "Unable to infer encoder feature dim; falling back to 2048. "
                    "If your backbone is not ResNet50, please ensure encoder exposes `out_dim`, `fc.in_features` or `classifier[-1].in_features`."
                )
            except Exception:
                pass
            num_features = 2048

        self.num_features = num_features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pre_allocate = num_classes
        self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)

        nn.init.orthogonal_(self.fc.weight)
        self.dummy_orthogonal_classifier = nn.Linear(
            self.num_features, max(0, self.pre_allocate - self.base_classes), bias=False
        )
        # freeze dummy classifier weights and initialize from corresponding rows in fc
        self.dummy_orthogonal_classifier.weight.requires_grad = False
        # copy matching rows from fc.weight (shape: [pre_allocate, num_features])
        if self.pre_allocate - self.base_classes > 0:
            self.dummy_orthogonal_classifier.weight.data.copy_(
                self.fc.weight.data[self.base_classes : self.pre_allocate, :]
            )

    def forward_metric(self, x):
        x = self.encode(x)
        if "cos" in self.mode:

            x1 = F.linear(
                F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1)
            )
            x = x1

            x = self.net_alice.temperature * x

        elif "dot" in self.mode:
            x = self.fc(x)
            x = self.net_alice.temperature * x
        return x

    def forpass_fc(self, x):
        x = self.encode(x)
        if "cos" in self.mode:

            x = F.linear(
                F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1)
            )
            x = self.net_alice.temperature * x

        elif "dot" in self.mode:
            x = self.fc(x)
            x = self.net_alice.temperature * x
        return x

    def encode(self, x):
        self.encoder(x)[0]
        x = self.encoder.features
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def get_features(self, data=None):
        if data is not None:
            self.encoder(data)[0]
        x = self.encoder.features
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def pre_encode(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)

        return x

    def post_encode(self, x):
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)

        if "cos" in self.mode:
            x = F.linear(
                F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1)
            )
            x = self.net_alice.temperature * x

        elif "dot" in self.mode:
            x = self.fc(x)
            x = self.net_alice.temperature * x

        return x

    def forward(self, input):
        encoder_feature = self.encode(input)
        wf = F.linear(
            F.normalize(encoder_feature, p=2, dim=1),
            F.normalize(self.fc.weight, p=2, dim=1),
        )
        return wf

    def update_fc(self, dataloader, class_list, session):
        datas = None
        labels = None
        for batch in dataloader:
            data = batch["data"].cuda()
            label = batch["label"].cuda()
            # data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            m = data.size()[0] // b
            # labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
            data = self.encode(data).detach()
            if datas is None:
                datas = data
                labels = label
            else:
                datas = torch.cat((datas, data))
                labels = torch.cat((labels, label))
        if self.net_alice.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list) * m, self.num_features, device="cuda"),
                requires_grad=True,
            )
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(datas, labels, class_list, m)

        # if 'ft' in self.net_alice.new_mode:  # further finetune
        #     self.update_fc_ft(new_fc,data,label,session)

    def update_fc_avg(self, data, labels, class_list, m):
        # Compute one prototype per absolute class index in class_list
        new_fc = []
        for class_index in class_list:
            # find samples belonging to this absolute class index
            data_index = (labels == class_index).nonzero().squeeze(-1)
            if data_index.numel() == 0:
                # no samples for this class in the batch; append zero vector
                proto = torch.zeros(self.num_features, device=data.device)
            else:
                embedding = data[data_index]
                proto = embedding.mean(0)
            new_fc.append(proto)
            # assign prototype to fc row corresponding to absolute class index
            if 0 <= class_index < self.pre_allocate:
                self.fc.weight.data[class_index] = proto
                # if dummy classifier exists and the index maps into it, update
                dummy_idx = class_index - self.base_classes
                if 0 <= dummy_idx < self.dummy_orthogonal_classifier.weight.data.size(0):
                    self.dummy_orthogonal_classifier.weight.data[dummy_idx] = proto

        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    def get_logits(self, x, fc):
        if "dot" in self.net_alice.new_mode:
            return F.linear(x, fc)
        elif "cos" in self.net_alice.new_mode:
            return self.net_alice.temperature * F.linear(
                F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1)
            )

    def update_fc_ft(self, new_fc, data, label, session):
        new_fc = new_fc.clone().detach()
        new_fc.requires_grad = True
        optimized_parameters = [{"params": new_fc}]
        optimizer = torch.optim.SGD(
            optimized_parameters,
            lr=self.args.optimizer.lr_new,
            momentum=0.9,
            dampening=0.9,
            weight_decay=0,
        )

        with torch.enable_grad():
            for epoch in range(self.args.optimizer.epochs_new):
                old_fc = self.fc.weight[
                    : self.args.dataloader.base_classes
                    + self.args.dataloader.way * (session - 1),
                    :,
                ].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data, fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[
            self.args.dataloader.base_classes
            + self.args.dataloader.way * (session - 1) : self.args.dataloader.base_class
            + self.args.dataloader.way * session,
            :,
        ].copy_(new_fc.data)
