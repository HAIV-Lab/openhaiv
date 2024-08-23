from ncdia.utils import TRAINERS
from .base import BaseTrainer
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import Hook
import os

@HOOKS.register
class PretrainHook(Hook):
    def __init__(self) -> None:
        super().__init__()

    def after_train(self, trainer) -> None:
        filename = 'task_' + str(trainer.session) + '.pth'
        trainer.save_ckpt(os.path.join(trainer.work_dir, filename))

@TRAINERS.register
class PreTrainer(BaseTrainer):
    """PreTrainer class for pre-training a model on session 0.

    """
    def __init__(self, *args, **kwargs):
        super(PreTrainer, self).__init__(*args, **kwargs)
        self.hook = PretrainHook()
        self.register_hook(self.hook)

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
        data = batch['data']            # data: (B, C, H, W) | list of (B, C, H, W)
        label = batch['label']          # label: (B,) | list of (B,)
        attribute = batch['attribute']  # attribute: (B, A) | list of (B, A)
        imgpath = batch['imgpath']      # imgpath: list(str) of length B
        return data, label, attribute, imgpath




