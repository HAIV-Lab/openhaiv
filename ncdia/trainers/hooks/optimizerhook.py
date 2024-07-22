from ncdia.utils import HOOKS
from .hook import Hook


@HOOKS.register()
class OptimizerHook(Hook):
    """A hook to put optimizer to zero_grad and step during training.
    """

    priority = 'NORMAL'

    def before_train_iter(self, trainer) -> None:
        """Put optimizer to zero_grad before training iteration.

        Args:
            trainer (BaseTrainer): Trainer object.
        """
        trainer.optimizer.zero_grad()

    def after_train_iter(self, trainer) -> None:
        """Put optimizer to step after training iteration.

        Args:
            trainer (BaseTrainer): Trainer object.
        """
        trainer.optimizer.step()
