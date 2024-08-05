import torch.nn as nn

from ncdia.utils import TRAINERS
from .pretrainer import PreTrainer


@TRAINERS.register
class IncTrainer(PreTrainer):
    """IncTrainer class for incremental training.

    Args:
        num_sess (int): Number of sessions. Default: 1.

    Attributes:
        num_sess (int): Number of sessions.
        session (int): Session number. If == 0, execute pre-training.
            If > 0, execute incremental training.

    """
    def __init__(
            self,
            num_sess: int = 1,
            *args, **kwargs
    ) -> None:
        super(IncTrainer, self).__init__(*args, **kwargs)
        self.num_sess = num_sess
        self._session = 0

    @property
    def session(self) -> int:
        """int: Session number. If == 0, execute pre-training.
        If > 0, execute incremental training."""
        return self._session

    def train(self) -> nn.Module:
        """Incremental training.
        `self.num_sess` determines the number of sessions,
        and session number is stored in `self.session`.

        Returns:
            model (nn.Module): Trained model.
        """
        for session in range(self.num_sess):
            self._session = session
            super(IncTrainer, self).train()
        return self.model
