import torch
import torch.nn as nn

from ncdia.utils import ALGORITHMS
from ncdia.utils.metrics import accuracy
from ncdia.algorithms.base import BaseAlg
from ncdia.algorithms.supervised.standard import StandardSL
from ncdia.trainers.hooks import AlgHook, QuantifyHook
import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
from .metrics import ood_metrics, search_threshold
from ncdia.trainers.hooks import AlgHook

@ALGORITHMS.register
class ENERGY(StandardSL):

    def __init__(self, trainer) -> None:
        super().__init__(trainer)

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
        outputs = model(data)

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
        train_gt: torch.Tensor,
        train_logits: torch.Tensor,
        train_feat: torch.Tensor,
        tpr_th: float = 0.95,
        prec_th: float = None,
    ):
        print("Energy inference..")
        ood_gt = -1 * np.ones(ood_logits.shape[0])

        id_conf = logsumexp(id_logits.cpu(), axis=-1)
        ood_conf = logsumexp(ood_logits.cpu(), axis=-1)

        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt.cpu(), ood_gt])

        if prec_th is None:
            return ood_metrics(conf, label, tpr_th), None
        else:
            return ood_metrics(conf, label, tpr_th), search_threshold(conf, label, prec_th)
