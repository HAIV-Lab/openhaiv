import os
from typing import Dict

from ncdia.utils import HOOKS, Logger
from .hook import Hook


@HOOKS.register()
class LoggerHook(Hook):
    """A hook to log information during training and evaluation.
    
    Args:
        interval (int): Logging interval every `interval` steps.
        ignore_last (bool): Whether to ignore the last step when
            the number of steps is not divisible by `interval`.
        exp_name (str): Experiment name to name the log file.
        out_dir (str): Output directory to save logs.
            If not specified, `trainer.work_dir` will be used.
        out_suffix (str): Suffix of the output log file,
            such as ".log" or ".txt".
        timestamp (bool): Whether to add timestamp to the log.
    """
    
    priority = 'BELOW_NORMAL'

    def __init__(
            self,
            interval: int = 10,
            ignore_last: bool = True,
            exp_name: str = 'exp',
            out_dir: str = None,
            out_suffix: str = '.log',
            timestamp: bool = True,
    ):
        super(LoggerHook, self).__init__()

        interval = int(interval)
        if interval <= 0:
            raise ValueError('interval must be a positive integer')
        
        exp_name = str(exp_name)
        out_suffix = str(out_suffix)

        self.interval = interval
        self.ignore_last = ignore_last
        self.exp_name = exp_name
        self.out_dir = out_dir
        self.out_suffix = out_suffix
        self.timestamp = timestamp
        self.logger = None
    
    def info(self, msg: str, **kwargs) -> None:
        """Log information.
        
        Args:
            msg (str): Information to be logged.

        Raises:
            RuntimeError: If logger is not initialized.
                logger is initialized in `before_run` method.
        """
        if not self.logger:
            raise RuntimeError('logger is not initialized')
        
        if self.timestamp:
            self.logger.info(msg, **kwargs)
        else:
            self.logger.write(msg, **kwargs)

    def before_run(self, trainer) -> None:
        """Log the start of training.
        
        Args:
            trainer (BaseTrainer): Trainer object.
        """
        if self.out_dir is None:
            self.out_dir = str(trainer.work_dir)
        else:
            self.out_dir = str(self.out_dir)

        self.log_file = os.path.join(
            self.out_dir, f'{self.exp_name}{self.out_suffix}')
        self.logger = Logger(self.log_file)

        # Create config file and save to disk in the form of yaml
        self.logger.create_config(trainer.cfg['cfg'])

        # Print config and save to log file
        self.logger.write(trainer.cfg)

        trainer.cfg.logger = self.logger

    def after_train_iter(
            self,
            trainer: object,
            batch_idx: int,
            data_batch: dict | tuple | list | None = None,
            outputs: dict | None = None
    ) -> None:
        """
        
        """

    def after_val_iter(
            self,
            trainer: object,
            batch_idx: int,
            data_batch: dict | tuple | list | None = None,
            outputs: os.Sequence | None = None
    ) -> None:
        """
        
        """
        
    def after_test_iter(
            self,
            trainer: object,
            batch_idx: int,
            data_batch: dict | tuple | list | None = None,
            outputs: os.Sequence | None = None
    ) -> None:
        """
        
        """
        
    def after_val_epoch(
            self,
            trainer: object,
            metrics: Dict[str, float] | None = None
    ) -> None:
        """
        
        """
        
    def after_test_epoch(
            self,
            trainer: object,
            metrics: Dict[str, float] | None = None
    ) -> None:
        """
        
        """
        