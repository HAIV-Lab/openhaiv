from ncdia.utils import HOOKS
from .hook import Hook


@HOOKS.register
class SchedulerHook(Hook):
    """A hook to change learning rate during training."""

    priority = "NORMAL"

    def init_trainer(self, trainer) -> None:
        """Initialize scheduler for trainer.

        Args:
            trainer (BaseTrainer): Trainer object.
        """
        trainer.scheduler

    def after_train_epoch(self, trainer) -> None:
        """Change learning rate after training epoch.

        Args:
            trainer (BaseTrainer): Trainer object.
        """
        trainer.scheduler.step()
