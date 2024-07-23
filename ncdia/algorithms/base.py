from ncdia.utils import ALGORITHMS
from ncdia.utils.cfg import Configs

@ALGORITHMS.register
class BaseAlg(object):
    """Basic algorithm class to define the interface of an algorithm.

    Args:
        trainer (object): Trainer object.

    Containing:
        - train_step(trainer, data, label, *args, **kwargs)
        - val_step(trainer, data, label, *args, **kwargs)
        - test_step(trainer, data, label, *args, **kwargs)

    """
    def __init__(self, trainer):
        super(BaseAlg, self).__init__()
        self.trainer = trainer
        self.cfg = trainer.cfg
        self.args = self.cfg

    def train_step(self, trainer, data, label, *args, **kwargs):
        """Training step.

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
                - other `key:value` pairs.
        """
        raise NotImplementedError

    def val_step(self, trainer, data, label, *args, **kwargs):
        """Validation step.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Validation results. Contains the following keys:
                - "loss": Loss value.
                - "acc": Accuracy value.
                - other `key:value` pairs.
        """
        raise NotImplementedError
    
    def test_step(self, trainer, data, label, *args, **kwargs):
        """Test step.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Test results. Contains the following keys:
                - "loss": Loss value.
                - "acc": Accuracy value.
                - other `key:value` pairs.
        """
        raise NotImplementedError
