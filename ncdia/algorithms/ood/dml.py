import torch
import torch.nn as nn

from ncdia.utils import ALGORITHMS
from ncdia.utils.metrics import accuracy
from ncdia.algorithms.base import BaseAlg
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook


@HOOKS.register
class DMLHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()

    def before_train(self, trainer) -> None:
        trainer.center_optimizer = torch.optim.SGD(trainer.criterion.parameters(), lr=0.5)

    def before_train_iter(self, trainer, batch_idx, data_batch) -> None:
        trainer.center_optimizer.zero_grad()
        epoch = trainer.epoch
        center_milestones = [0, 60, 80]
        assigned_center_weights = [0.0, 0.001, 0.005]
        center_weight = assigned_center_weights[0]
        for i, ms in enumerate(center_milestones):
            if epoch >= ms:
                center_weight = assigned_center_weights[i]
        trainer.center_weight = center_weight

    def after_train_iter(self, trainer, batch_idx, data_batch, outputs) -> None:
        center_weight = trainer.center_weight
        for param in trainer.criterion.parameters():
            param.grad.data *= (1./(center_weight + 1e-12))
        trainer.center_optimizer.step()


@ALGORITHMS.register
class DML(BaseAlg):
    """Standard supervised learning algorithm.

    Containing:
        - train_step(trainer, data, label, *args, **kwargs)
        - val_step(trainer, data, label, *args, **kwargs)
        - test_step(trainer, data, label, *args, **kwargs)

    """
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.trainer = trainer

        if trainer.model.loss == 'center':
            hook = DMLHook()
            trainer.register_hook(hook)

    def train_step(self, trainer, data, label, *args, **kwargs):
        """Training step for standard supervised learning.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Training results. Contains the following keys:
                - "loss": Loss value.
                - "acc": Accuracy value.
        """
        model = trainer.model
        criterion = trainer.criterion
        device = trainer.device

        data, label = data.to(device), label.to(device)
        outputs = model(data)
        if model.loss == 'center':
            features = model.get_features()

            loss_ct = criterion(features, label)
            loss_func = nn.CrossEntropyLoss()
            loss_ce = loss_func(outputs, label)

            loss = loss_ce + loss_ct * trainer.center_weight
        else:
            loss = criterion(outputs, label)
        
        acc = accuracy(outputs, label)[0]

        loss.backward()
        return {"loss": loss.item(), "acc": acc.item()}

    def val_step(self, trainer, data, label, *args, **kwargs):
        """Validation step for standard supervised learning.

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
        """Test step for standard supervised learning.

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
