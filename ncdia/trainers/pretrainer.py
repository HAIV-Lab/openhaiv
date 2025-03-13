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
    def __init__(
            self,
            max_epochs: int = 1,
            **kwargs
    ) -> None:
        super(PreTrainer, self).__init__(**kwargs)
        self._max_epochs = max_epochs

    def train_step(self, batch, **kwargs):
        """Training step.

        Args:
            batch (dict | tuple | list): A batch of data.

        Returns:
            results (dict): Training result.
        """
        data, label, attribute, imgpath = self.batch_parser(batch)
        return self.algorithm.train_step(self, data, label, attribute, imgpath)

    def val_step(self, batch, **kwargs):
        """Validation step.

        Args:
            batch (dict | tuple | list): A batch of data.

        Returns:
            results (dict): Validation result.
        """
        data, label, attribute, imgpath = self.batch_parser(batch)
        return self.algorithm.val_step(self, data, label, attribute, imgpath)

    def test_step(self, batch, **kwargs):
        """Test step.

        Args:
            batch (dict | tuple | list): A batch of data.
        
        Returns:
            results (dict): Test result.
        """
        data, label, attribute, imgpath = self.batch_parser(batch)
        return self.algorithm.test_step(self, data, label, attribute, imgpath)

    @staticmethod
    def batch_parser(batch):
        """Parse a batch of data.

        Args:
            batch (dict | tuple | list): A batch of data.

        Returns:
            data (torch.Tensor | list): Input data.
            label (torch.Tensor | list): Label data.
            attribute (torch.Tensor | list): Attribute data.
            imgpath (list of str): Image path.
        """
        if isinstance(batch, dict):
            data = batch['data']            # data: (B, C, H, W) | list of (B, C, H, W)
            label = batch['label']          # label: (B,) | list of (B,)
            attribute = batch['attribute']  # attribute: (B, A) | list of (B, A)
            imgpath = batch['imgpath']      # imgpath: list(str) of length B
        elif isinstance(batch, (tuple, list)):
            # Assume the batch is a tuple or list of (data, label, attribute, imgpath)
            data = batch[1]
            label = batch[2]
            attribute = []
            imgpath = []
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
        
        return data, label, attribute, imgpath