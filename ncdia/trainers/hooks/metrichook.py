from copy import deepcopy
from ncdia.utils import HOOKS, METRICS
from .hook import Hook


@HOOKS.register
class MetricHook(Hook):
    """A hook to calculate metrics during evaluation and testing.

    Args:
        metrics (dict | None): Metrics to be calculated.
            Each key-value pair is a metric name and its parameters.
            If None, default metrics will be used.

    Example:
        >>> metrics = {
        >>>     "acc": {"type": "average"},
        >>>     "loss": {"type": "average"},
        >>> }
        >>> hook = MetricHook(metrics)

    """

    priority = "NORMAL"
    DEFAULT_METRICS = dict(
        {
            "acc": {"type": "average"},
            "loss": {"type": "average"},
        }
    )

    def __init__(
        self,
        metrics: dict | None = None,
    ):
        super(MetricHook, self).__init__()
        if metrics is None:
            metrics = deepcopy(self.DEFAULT_METRICS)

        if not isinstance(metrics, dict):
            raise TypeError(f"Metrics {metrics} is not a dict.")

        self.metrics = metrics
        for key, value in metrics.items():
            self.metrics[key] = METRICS.build(value)

    def reset_metrics(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()

    def update_metrics(self, outputs: dict | None, n: int = 1):
        """Update all metrics.

        Args:
            outputs (dict): Model outputs.
            n (int): Number of samples.
        """
        if not isinstance(outputs, dict):
            return

        for key, value in outputs.items():
            if key in self.metrics:
                self.metrics[key].update(value, n)

    def before_val(self, trainer) -> None:
        """Reset metrics before validation.

        Args:
            trainer (BaseTrainer): Trainer object.
        """
        self.reset_metrics()
        trainer._metrics = self.metrics

    def before_test(self, trainer) -> None:
        """Reset metrics before testing.

        Args:
            trainer (BaseTrainer): Trainer object.
        """
        self.reset_metrics()
        trainer._metrics = self.metrics

    def after_val_iter(
        self, trainer, batch_idx: int, data_batch=None, outputs=None
    ) -> None:
        """Update metrics after validation iteration.

        Args:
            trainer (BaseTrainer): Trainer object.
            batch_idx (int): Index of the current batch.
            data_batch (dict | tuple | list): Data batch.
            outputs (dict): Model outputs.
        """
        self.update_metrics(outputs, len(data_batch))
        trainer._metrics = self.metrics

    def after_test_iter(
        self, trainer, batch_idx: int, data_batch=None, outputs=None
    ) -> None:
        """Update metrics after test iteration.

        Args:
            trainer (BaseTrainer): Trainer object.
            batch_idx (int): Index of the current batch.
            data_batch (dict | tuple | list): Data batch.
            outputs (dict): Model outputs.
        """
        self.update_metrics(outputs, len(data_batch))
        trainer._metrics = self.metrics
