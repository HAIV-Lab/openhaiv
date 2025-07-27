import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from ncdia.utils import ALGORITHMS
from ncdia.utils.metrics import accuracy
from ncdia.algorithms.base import BaseAlg
import torch.nn.functional as F
import numpy as np
from ncdia.trainers.hooks import QuantifyHook
from ncdia.utils import HOOKS
from .metrics import ood_metrics, search_threshold


@HOOKS.register
class VOSHook(QuantifyHook):
    def __init__(self) -> None:
        super().__init__()

    def after_test(self, trainer) -> None:
        pass


@ALGORITHMS.register
class VOS(BaseAlg):
    """ODIN

    Args:
        id_gt (torch.Tensor): ID ground truth labels. Shape (N,).
        id_logits (torch.Tensor): ID logits. Shape (N, C).
        id_feat (torch.Tensor): ID features. Shape (N, D).
        ood_gt (torch.Tensor): OOD ground truth labels. Shape (M,).
        ood_logits (torch.Tensor): OOD logits. Shape (M, C).
        ood_feat (torch.Tensor): OOD features. Shape (M, D).
        train_logits (torch.Tensor): Training logits. Shape (K, C).
        train_feat (torch.Tensor): Training features. Shape (K, D).
        tpr_th (float): True positive rate threshold to compute
            false positive rate. Default is 0.95.
        prec_th (float | None): Precision threshold for searching threshold.
            If None, not searching for threshold. Default is None.

    Returns:
        fpr (float): False positive rate.
        auroc (float): Area under the ROC curve.
        aupr_in (float): Area under the precision-recall curve
            for in-distribution samples.
        aupr_out (float): Area under the precision-recall curve
            for out-of-distribution
    """

    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        hook = VOSHook()
        trainer.register_hook(hook)
        self.hyparameters = None

        self.config = trainer.cfg
        self.net = trainer.model

        weight_energy = torch.nn.Linear(
            self.config.model.network["num_classes"], 1
        ).cuda()
        torch.nn.init.uniform_(weight_energy.weight)

        self.logistic_regression = torch.nn.Linear(1, 2).cuda()
        self.number_dict = {}

        for i in range(self.config.model.network["num_classes"]):
            self.number_dict[i] = 0
        self.data_dict = torch.zeros(
            self.config.model.network["num_classes"],
            self.config.sample_number,
            self.config.feature_dim,
        ).cuda()
        self.num_classes = self.config.model.network["num_classes"]
        self.eye_matrix = torch.eye(self.config.feature_dim, device="cuda")

        self.sample_number = self.config.sample_number

    def train_step(self, trainer, data, label, attribute, imgpath):
        """Validation step for Decoupling MaxLogit.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Validation results. Contains the following:
                - "loss": Loss value.
                - "acc": Accuracy value.
        """
        model = trainer.model
        device = trainer.device
        num_classes = self.num_classes

        data, label = data.to(device), label.to(device)

        x, outputs = model.forward_feature(data)

        sum_temp = 0
        for index in range(num_classes):
            sum_temp += self.number_dict[index]
        lr_reg_loss = torch.zeros(1).cuda()[0]

        if (
            sum_temp == num_classes * self.sample_number
            and trainer.epoch < self.config.start_epoch
        ):
            target_numpy = label.cpu().data.numpy()
            for index in range(len(label)):
                dict_key = target_numpy[index]
                self.data_dict[dict_key] = torch.cat(
                    (self.data_dict[dict_key][1:], outputs[index].detach().view(1, -1)),
                    0,
                )

        elif (
            sum_temp == num_classes * self.sample_number
            and trainer.epoch >= self.config.start_epoch
        ):
            target_numpy = label.cpu().data.numpy()
            for index in range(len(label)):
                dict_key = target_numpy[index]
                self.data_dict[dict_key] = torch.cat(
                    (self.data_dict[dict_key][1:], outputs[index].detach().view(1, -1)),
                    0,
                )

            for index in range(num_classes):
                if index == 0:
                    X = self.data_dict[index] - self.data_dict[index].mean(0)
                    mean_embed_id = self.data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat(
                        (X, self.data_dict[index] - self.data_dict[index].mean(0)), 0
                    )
                    mean_embed_id = torch.cat(
                        (mean_embed_id, self.data_dict[index].mean(0).view(1, -1)), 0
                    )

            temp_precision = torch.mm(X.t(), X) / len(X)
            temp_precision += 0.0001 * self.eye_matrix
            for index in range(num_classes):
                new_dis = MultivariateNormal(
                    loc=mean_embed_id[index], covariance_matrix=temp_precision
                )
                negative_samples = new_dis.rsample((self.config.sample_from,))
                prob_density = new_dis.log_prob(negative_samples)
                cur_samples, index_prob = torch.topk(-prob_density, self.config.select)
                if index == 0:
                    ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat(
                        (ood_samples, negative_samples[index_prob]), 0
                    )
            if len(ood_samples) != 0:

                energy_score_for_fg = log_sum_exp(x, num_classes=num_classes, dim=1)
                try:
                    predictions_ood = self.net.network.fc(ood_samples)
                except AttributeError:
                    predictions_ood = self.net.network.fc(ood_samples)

                energy_score_for_bg = log_sum_exp(
                    predictions_ood, num_classes=num_classes, dim=1
                )

                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                labels_for_lr = torch.cat(
                    (
                        torch.ones(len(outputs)).cuda(),
                        torch.zeros(len(ood_samples)).cuda(),
                    ),
                    -1,
                )

                output1 = self.logistic_regression(input_for_lr.view(-1, 1))

                lr_reg_loss = F.cross_entropy(output1, labels_for_lr.long())
        else:
            target_numpy = label.cpu().data.numpy()
            for index in range(len(label)):
                dict_key = target_numpy[index]

                if self.number_dict[dict_key] < self.sample_number:
                    self.data_dict[dict_key][self.number_dict[dict_key]] = outputs[
                        index
                    ].detach()
                    self.number_dict[dict_key] += 1
        loss = F.cross_entropy(x, label)
        loss += self.config.loss_weight * lr_reg_loss

        loss.backward()
        acc = accuracy(x, label)[0]
        return {"loss": loss.item(), "acc": acc.item()}

    def val_step(self, trainer, data, label, *args, **kwargs):
        """Validation step for Decoupling MaxLogit.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Validation results. Contains the following:
                - "loss": Loss value.
                - "acc": Accuracy value.
        """
        model = trainer.model
        device = trainer.device

        data, label = data.to(device), label.to(device)
        x, outputs = model.forward_feature(data)

        criterion = nn.CrossEntropyLoss()

        loss = criterion(outputs, label)
        acc = accuracy(outputs, label)[0]

        return {"loss": loss.item(), "acc": acc.item()}

    def test_step(self, trainer, data, label, *args, **kwargs):
        """Test step for Decoupling MaxLogit.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Test results. Contains the following:
                - "loss": Loss value.
                - "acc": Accuracy value.
        """
        return self.val_step(trainer, data, label, *args, **kwargs)

    @staticmethod
    def eval(
        id_gt: torch.Tensor,
        id_logits: torch.Tensor,
        id_feat: torch.Tensor,
        ood_logits: torch.Tensor,
        ood_feat: torch.Tensor,
        train_logits: torch.Tensor = None,
        train_feat: torch.Tensor = None,
        train_gt: torch.Tensor = None,
        tpr_th: float = 0.95,
        prec_th: float = None,
        hyparameters: dict = None,
        id_local_logits=None,
        id_local_feat=None,
        ood_local_logits=None,
        ood_local_feat=None,
        train_local_logits=None,
        train_local_feat=None,
        prototypes=None,
        s_prototypes=None,
        hyperparameters=None,
    ):
        """ODIN"""
        temperature = 1

        neg_ood_gt = -1 * np.ones(ood_logits.shape[0])

        id_conf = temperature * torch.logsumexp(id_logits / temperature, dim=1)
        ood_conf = temperature * torch.logsumexp(ood_logits / temperature, dim=1)

        conf = np.concatenate([id_conf.cpu(), ood_conf.cpu()])
        label = np.concatenate([id_conf, neg_ood_gt])
        # label = np.concatenate([np.ones_like(id_conf), neg_ood_gt])
        # neg_label = np.concatenate([np.ones_like(id_conf), neg_ood_gt])

        if prec_th is None:
            return conf, label, *ood_metrics(conf, label, tpr_th), None, None, None
        else:
            return (
                conf,
                label,
                *ood_metrics(conf, label, tpr_th),
                *search_threshold(conf, label, prec_th),
            )


def log_sum_exp(value, num_classes=10, dim=None, keepdim=False):
    """Numerically stable implementation of the operation."""
    value.exp().sum(dim, keepdim).log()

    # TODO: torch.max(value, dim=None) threw an error at time of writing
    weight_energy = torch.nn.Linear(num_classes, 1).cuda()
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)

        output = m + torch.log(
            torch.sum(
                F.relu(weight_energy.weight) * torch.exp(value0),
                dim=dim,
                keepdim=keepdim,
            )
        )
        # set lower bound
        out_list = output.cpu().detach().numpy().tolist()
        for i in range(len(out_list)):
            if out_list[i] < -1:
                out_list[i] = -1
            else:
                continue
        output = torch.Tensor(out_list).cuda()
        return output
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        # if isinstance(sum_exp, Number):
        #     return m + math.log(sum_exp)
        # else:
        return m + torch.log(sum_exp)
