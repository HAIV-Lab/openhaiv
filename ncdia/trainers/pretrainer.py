import torch

from ncdia.utils import TRAINERS
from .base import BaseTrainer


@TRAINERS.register()
class PreTrainer(BaseTrainer):
    """PreTrainer class for pre-training a model on session 0.

    Args:
    
    
    """
    def __init__(self, *args, **kwargs):
        super(PreTrainer, self).__init__(*args, **kwargs)

    def train_step(self, batch, **kwargs):
        """Training step.

        Args:
            batch (dict | tuple | list): A batch of data.
        """
        data, label, attribute, imgpath = self.batch_parser(batch)

    def eval_step(self, batch, **kwargs):
        """Evaluation step.

        Args:
            batch (dict | tuple | list): A batch of data.
        """
        data, label, attribute, imgpath = self.batch_parser(batch)

    def test_step(self, batch, **kwargs):
        """Test step.

        Args:
            batch (dict | tuple | list): A batch of data.
        """
        data, label, attribute, imgpath = self.batch_parser(batch)

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
        data = batch['data']            # data: (B, C, H, W) | list of (B, C, H, W)
        label = batch['label']          # label: (B,) | list of (B,)
        attribute = batch['attribute']  # attribute: (B, A) | list of (B, A)
        imgpath = batch['imgpath']      # imgpath: list(str) of length B
        return data, label, attribute, imgpath
    