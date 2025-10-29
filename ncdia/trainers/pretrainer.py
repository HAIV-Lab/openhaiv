from ncdia.utils import TRAINERS
from .base import BaseTrainer


@TRAINERS.register
class PreTrainer(BaseTrainer):
    """PreTrainer class for pre-training a model on session 0.

    Args:
        max_epochs (int): Maximum number of epochs. Default: 1.

    Attributes:
        max_epochs (int): Total epochs for training.

    """

    def __init__(self, max_epochs: int = 1, **kwargs) -> None:
        super(PreTrainer, self).__init__(**kwargs)
        self._max_epochs = max_epochs

    def train_step(self, batch, **kwargs):
        """Training step.

        Args:
            batch (dict | tuple | list): A batch of data.

        Returns:
            results (dict): Training result.
        """

        data, label, imgpath = self.batch_parser(batch)
        return self.algorithm.train_step(self, data, label, imgpath)

    def val_step(self, batch, **kwargs):
        """Validation step.

        Args:
            batch (dict | tuple | list): A batch of data.

        Returns:
            results (dict): Validation result.
        """
        data, label, imgpath = self.batch_parser(batch)
        return self.algorithm.val_step(self, data, label, imgpath)

    def test_step(self, batch, **kwargs):
        """Test step.

        Args:
            batch (dict | tuple | list): A batch of data.

        Returns:
            results (dict): Test result.
        """
        data, label, imgpath = self.batch_parser(batch)
        return self.algorithm.test_step(self, data, label, imgpath)

    @staticmethod
    def batch_parser(batch):
        """Parse a batch of data.

        Args:
            batch (dict | tuple | list): A batch of data.

        Returns:
            data (torch.Tensor | list): Input data.
            label (torch.Tensor | list): Label data.
            imgpath (list of str): Image path.
        """
        data = batch["data"]  # data: (B, C, H, W) | list of (B, C, H, W)
        label = batch["label"]  # label: (B,) | list of (B,)
        imgpath = batch["imgpath"]  # imgpath: list(str) of length B
        return data, label, imgpath
