import os
from typing import Dict, Sequence

from ncdia.utils import HOOKS, Logger
from .hook import Hook


@HOOKS.register
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

    priority = "BELOW_NORMAL"

    def __init__(
        self,
        interval: int = 25,
        ignore_last: bool = False,
        exp_name: str = "exp",
        out_dir: str = None,
        out_suffix: str = ".log",
        timestamp: bool = True,
    ):
        super(LoggerHook, self).__init__()

        interval = int(interval)
        if interval <= 0:
            raise ValueError("interval must be a positive integer")

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
            raise RuntimeError("logger is not initialized")

        if self.timestamp:
            self.logger.info(msg, **kwargs)
        else:
            self.logger.write(msg, **kwargs)

    def init_trainer(self, trainer) -> None:
        """Create logger and save to trainer.

        Args:
            trainer (BaseTrainer): Trainer object.
        """
        if self.out_dir is None:
            self.out_dir = str(trainer.work_dir)
        else:
            self.out_dir = str(self.out_dir)

        self.log_file = os.path.join(self.out_dir, f"{self.exp_name}{self.out_suffix}")
        self.logger = Logger(self.log_file)

        # Create config file and save to disk in the form of yaml
        self.logger.create_config(trainer.cfg["cfg"])

        # Print config and save to log file
        self.logger.write(trainer.cfg)

        trainer._logger = self.logger

    def before_run(self, trainer) -> None:
        """ """

    def after_train_iter(
        self,
        trainer: object,
        batch_idx: int,
        data_batch: dict | tuple | list | None = None,
        outputs: dict | None = None,
    ) -> None:
        """Print output information every `interval` train steps.

        Args:
            trainer (object): Trainer object.
            batch_idx (int): Index of the current batch.
            data_batch (dict | tuple | list): Data batch.
            outputs (dict): Output results.
        """
        if self.interval <= 0:
            return

        if self.ignore_last and batch_idx == trainer.max_train_iters - 1:
            return

        if (batch_idx + 1) % self.interval == 0:
            msg = (
                f"Epoch(train) [{trainer.epoch + 1}/{trainer.max_epochs}] "
                f"Iter [{batch_idx + 1}/{trainer.max_train_iters}] "
            )
            if outputs:
                for key, value in outputs.items():
                    if isinstance(value, float):
                        msg += f"| {key}: {value:.4f} "
                    elif isinstance(value, (int, str)):
                        msg += f"| {key}: {value} "
            self.info(msg)

    def after_val_iter(
        self,
        trainer: object,
        batch_idx: int,
        data_batch: dict | tuple | list | None = None,
        outputs: Sequence | None = None,
    ) -> None:
        """Print output information every `interval` validation steps.

        Args:
            trainer (object): Trainer object.
            batch_idx (int): Index of the current batch.
            data_batch (dict | tuple | list): Data batch.
            outputs (Sequence): Output results.
        """
        if self.interval <= 0:
            return

        if self.ignore_last and batch_idx == trainer.max_val_iters - 1:
            return

        if (batch_idx + 1) % self.interval == 0:
            msg = f"Epoch(val) Iter [{batch_idx + 1}/{trainer.max_val_iters}] "
            if outputs:
                for key, value in outputs.items():
                    if isinstance(value, float):
                        msg += f"| {key}: {value:.4f} "
                    elif isinstance(value, (int, str)):
                        msg += f"| {key}: {value} "
            self.info(msg)

    def after_test_iter(
        self,
        trainer: object,
        batch_idx: int,
        data_batch: dict | tuple | list | None = None,
        outputs: Sequence | None = None,
    ) -> None:
        """Print output information every `interval` test steps.

        Args:
            trainer (object): Trainer object.
            batch_idx (int): Index of the current batch.
            data_batch (dict | tuple | list): Data batch.
            outputs (Sequence): Output results.
        """
        if self.interval <= 0:
            return

        if self.ignore_last and batch_idx == trainer.max_test_iters - 1:
            return

        if (batch_idx + 1) % self.interval == 0:
            msg = f"Epoch(test) Iter [{batch_idx + 1}/{trainer.max_test_iters}] "
            if outputs:
                for key, value in outputs.items():
                    if isinstance(value, float):
                        msg += f"| {key}: {value:.4f} "
                    elif isinstance(value, (int, str)):
                        msg += f"| {key}: {value} "
            self.info(msg)

    def before_val_epoch(self, trainer) -> None:
        """Print information before each validation epoch.

        Args:
            trainer (object): Trainer object.
        """
        self.info("Evaluating...")

    def after_val_epoch(
        self, trainer: object, metrics: Dict[str, float] | None = None
    ) -> None:
        """Print evaluation results after each validation epoch.

        Args:
            trainer (object): Trainer object.
            metrics (dict): Evaluation metrics.
        """
        if not isinstance(metrics, dict) or not metrics:
            return

        msg = f"Epoch(val) "
        for name, metric in metrics.items():
            value = metric.value
            if isinstance(value, float):
                msg += f"| {name}: {value:.4f} "
            elif isinstance(value, (int, str)):
                msg += f"| {name}: {value} "
        self.info(msg)

    def before_test_epoch(self, trainer) -> None:
        """Print information before each test epoch.

        Args:
            trainer (object): Trainer object.
        """
        self.info("Evaluating...")

    def after_test_epoch(
        self, trainer: object, metrics: Dict[str, float] | None = None
    ) -> None:
        """Print evaluation results after each test epoch.

        Args:
            trainer (object): Trainer object.
            metrics (dict): Evaluation metrics.
        """
        if not isinstance(metrics, dict) or not metrics:
            return

        msg = f"Epoch(test) "
        for name, metric in metrics.items():
            value = metric.value
            if isinstance(value, float):
                msg += f"| {name}: {value:.4f} "
            elif isinstance(value, (int, str)):
                msg += f"| {name}: {value} "
        self.info(msg)
