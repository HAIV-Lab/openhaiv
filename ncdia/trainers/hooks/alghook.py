from ncdia.utils import HOOKS
from .hook import Hook


@HOOKS.register
class AlgHook(Hook):
    """A hook to modify algorithm state in the pipeline.
    This class is a base class for all algorithm hooks.
    """

    priority = "NORMAL"

    def init_trainer(self, trainer) -> None:
        """Initialize algorithm for trainer.

        Args:
            trainer (BaseTrainer): Trainer object.
        """
        trainer.algorithm
