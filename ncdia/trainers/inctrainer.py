import torch.nn as nn

from ncdia.utils import TRAINERS, Configs
from .pretrainer import PreTrainer


@TRAINERS.register
class IncTrainer(PreTrainer):
    """IncTrainer class for incremental training.

    Args:
        sess_cfg (Configs): Session configuration.

    Attributes:
        sess_cfg (Configs): Session configuration.
        num_sess (int): Number of sessions.
        session (int): Session number. If == 0, execute pre-training.
            If > 0, execute incremental training.

    """
    def __init__(
            self,
            sess_cfg: Configs,
            *args, **kwargs
    ) -> None:
        super(IncTrainer, self).__init__(*args, **kwargs)
        self.sess_cfg = sess_cfg
        self.num_sess = len(sess_cfg.keys())
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

            cfg = self.sess_cfg[f's{session}'].cfg
            self._cfg.merge_from_config(cfg)
            self._cfg.freeze()

            self._session = session
            self._max_epochs = self._cfg.trainer.max_epochs or 1

            super(IncTrainer, self).__init__(cfg=self._cfg)

            super(IncTrainer, self).train()
        
        return self.model
