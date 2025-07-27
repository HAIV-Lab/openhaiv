import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncdia.utils import MODELS


@MODELS.register
class FACTNET(nn.Module):

    def __init__(
        self, network, base_classes, num_classes, att_classes, net_fact, mode="ft_cos"
    ):
        super().__init__()

        self.mode = mode
        self.base_classes = base_classes
        self.num_classes = num_classes
        self.att_classes = att_classes
        self.net_fact = net_fact
        self.network = network

        # pretrained=True follow TOPIC, models for cub is imagenet pre-trained.
        # https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
        network = network.cfg
        network["pretrained"] = True
        network["num_classes"] = 1000
        self.encoder = MODELS.build(network)

        self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pre_allocate = self.num_classes
        self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)

        nn.init.orthogonal_(self.fc.weight)
        self.dummy_orthogonal_classifier = nn.Linear(
            self.num_features, self.pre_allocate - self.base_classes, bias=False
        )
        self.dummy_orthogonal_classifier.weight.requires_grad = False

        self.dummy_orthogonal_classifier.weight.data = self.fc.weight.data[
            self.base_classes :, :
        ]
        print(self.dummy_orthogonal_classifier.weight.data.size())

        print("self.dummy_orthogonal_classifier.weight initialized over.")

    def forward_metric(self, x):
        x = self.encode(x)
        if "cos" in self.mode:

            x1 = F.linear(
                F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1)
            )
            x = x1
            x = self.net_fact.temperature * x

        elif "dot" in self.mode:
            x = self.fc(x)
            x = self.net_fact.temperature * x
        return x

    def forpass_fc(self, x):
        x = self.encode(x)
        if "cos" in self.mode:

            x = F.linear(
                F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1)
            )
            x = self.net_fact.temperature * x

        elif "dot" in self.mode:
            x = self.fc(x)
            x = self.net_fact.temperature * x
        return x

    def encode(self, x):
        self.encoder(x)[0]
        x = self.encoder.features
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def get_features(self, data=None):

        if data is not None:
            self.encoder(data)
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
            x = self.net_fact.temperature * x

        elif "dot" in self.mode:
            x = self.fc(x)
            x = self.net_fact.temperature * x

        return x

    def forward(self, input):
        if self.mode != "encoder":
            input = self.forward_metric(input)
            return input
        elif self.mode == "encoder":
            input = self.encode(input)
            return input
        else:
            raise ValueError("Unknown mode")

    def update_fc(self, dataloader, class_list, session):
        for batch in dataloader:
            data = batch["data"].cuda()
            label = batch["label"].cuda()
            # data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            m = data.size()[0] // b
            labels = torch.stack([label * m + ii for ii in range(m)], 1).view(-1)
            data = self.encode(data).detach()

        if self.net_fact.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list) * m, self.num_features, device="cuda"),
                requires_grad=True,
            )
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list, m)

        if "ft" in self.net_fact.new_mode:  # further finetune
            self.update_fc_ft(new_fc, data, label, session)

    def update_fc_avg(self, data, labels, class_list, m):
        new_fc = []
        for class_index in class_list:
            for i in range(m):
                index = class_index * m + i
                data_index = (labels == index).nonzero().squeeze(-1)
                embedding = data[data_index]
                proto = embedding.mean(0)
                new_fc.append(proto)
                self.fc.weight.data[index] = proto
                self.dummy_orthogonal_classifier.weight.data[index - self.base_classes]
        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    def get_logits(self, x, fc):
        if "dot" in self.net_fact.new_mode:
            return F.linear(x, fc)
        elif "cos" in self.net_fact.new_mode:
            return self.net_fact.temperature * F.linear(
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
            + self.args.dataloader.way
            * (session - 1) : self.args.dataloader.base_classes
            + self.args.dataloader.way * session,
            :,
        ].copy_(new_fc.data)
