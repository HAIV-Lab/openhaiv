import torch.nn as nn

from ncdia.utils import TRAINERS, Configs
from .pretrainer import PreTrainer


@TRAINERS.register
class IncTrainer(PreTrainer):
    """IncTrainer class for incremental training.

    Args:
        model (nn.Module): Model to be trained.
        cfg (dict, optional): Configuration for trainer.
        sess_cfg (Configs): Session configuration.
        session (int, optional): Session number. Default: 0.

    Attributes:
        sess_cfg (Configs): Session configuration.
        num_sess (int): Number of sessions.
        session (int): Session number. If == 0, execute pre-training.
            If > 0, execute incremental training.

    """
    def __init__(
            self,
            model: nn.Module = None,
            cfg: dict | None = None,
            sess_cfg: Configs | None = None,
            session: int = 0,
            **kwargs
    ) -> None:
        self.sess_cfg = sess_cfg
        self.num_sess = len(sess_cfg.keys())
        self.kwargs = kwargs

        s_cfg = sess_cfg[f's{session}'].cfg
        cfg.merge_from_config(s_cfg)
        cfg.freeze()

        super(IncTrainer, self).__init__(
            model=model,
            cfg=cfg,
            session=session,
            max_epochs=cfg.trainer.max_epochs or 1,
            **self.kwargs
        )

    def train(self) -> nn.Module:
        """Incremental training.
        `self.num_sess` determines the number of sessions,
        and session number is stored in `self.session`.

        Returns:
            model (nn.Module): Trained model.
        """
        for session in range(self.num_sess):

            if session > 0:
                new_instance = IncTrainer(
                    model=self.model,
                    cfg=self.cfg,
                    sess_cfg=self.sess_cfg,
                    session=session,
                    **self.kwargs
                )
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__

            super(IncTrainer, self).train()
        
        return self.model
