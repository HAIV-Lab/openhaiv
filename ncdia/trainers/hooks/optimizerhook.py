from ncdia.utils import HOOKS
from .hook import Hook


@HOOKS.register
class OptimizerHook(Hook):
    """A hook to put optimizer to zero_grad and step during training."""

    priority = "NORMAL"

    def init_trainer(self, trainer) -> None:
        """Initialize optimizer for trainer.

        Args:
            trainer (BaseTrainer): Trainer object.
        """
        trainer.optimizer

    def before_train_iter(self, trainer, batch_idx: int, data_batch=None) -> None:
        """Put optimizer to zero_grad before training iteration.

        Args:
            trainer (BaseTrainer): Trainer object.
            batch_idx (int): Batch index.
            data_batch (dict | tuple | list): A batch of data.
        """
        trainer.optimizer.zero_grad()

    def after_train_iter(
        self, trainer, batch_idx: int, data_batch=None, outputs=None
    ) -> None:
        """Put optimizer to step after training iteration.

        Args:
            trainer (BaseTrainer): Trainer object.
            batch_idx (int): Batch index.
            data_batch (dict | tuple | list): A batch of data.
            outputs (dict): Output of the model.
        """
        trainer.optimizer.step()
