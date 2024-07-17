from ncdia.utils import HOOKS
from .hook import Hook


@HOOKS.register()
class ModelHook(Hook):
    """A hook to change model state in the pipeline, 
    such as setting device, changing model to eval mode, etc.
    """

    priority = 'HIGHEST'

    def before_run(self, trainer) -> None:
        """Set model to device before running.

        Args:
            trainer (BaseTrainer): Trainer object.
        """
        trainer.model.to(trainer.device)

    def before_train_epoch(self, trainer) -> None:
        """Set model to train mode before training epoch.
        
        Args:
            trainer (BaseTrainer): Trainer object.
        """
        trainer.model.train()
        
    def before_val_epoch(self, trainer) -> None:
        """Set model to eval mode before validation epoch.

        Args:
            trainer (BaseTrainer): Trainer object.
        """
        trainer.model.eval()
    
    def before_test_epoch(self, trainer) -> None:
        """Set model to eval mode before testing epoch.

        Args:
            trainer (BaseTrainer): Trainer object.
        """
        trainer.model.eval()
