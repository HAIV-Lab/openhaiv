from ncdia.utils import ALGORITHMS
from ncdia.utils.metrics import accuracy
from ncdia.algorithms.base import BaseAlg


@ALGORITHMS.register
class StandardSL(BaseAlg):
    """Standard supervised learning algorithm.

    Containing:
        - train_step(trainer, data, label, *args, **kwargs)
        - val_step(trainer, data, label, *args, **kwargs)
        - test_step(trainer, data, label, *args, **kwargs)

    """
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

        if isinstance(data, dict):
            if len(data) == 2:
                data_a, data_b, label = data["a"].to(device), data["b"].to(device), label.to(device)
                outputs = model((data_a, data_b))
        else:
            data, label = data.to(device), label.to(device)
            outputs = model(data)

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
        criterion = trainer.criterion
        device = trainer.device

        if isinstance(data, dict):
            if len(data) == 2:
                data_a, data_b, label = data["a"].to(device), data["b"].to(device), label.to(device)
                outputs = model((data_a, data_b))
        else:
            data, label = data.to(device), label.to(device)
            outputs = model(data)

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
