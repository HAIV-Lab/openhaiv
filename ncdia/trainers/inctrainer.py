import torch.nn as nn

from ncdia.utils import TRAINERS, Configs
from ncdia.dataloader import MergedDataset
from .pretrainer import PreTrainer
from .hooks import NCDHook


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
        hist_trainset (MergedDataset): Historical training dataset.
        hist_valset (MergedDataset): Historical validation dataset.
        hist_testset (MergedDataset): Historical testing dataset.

    """
    def __init__(
            self,
            cfg: dict | None = None,
            sess_cfg: Configs | None = None,
            ncd_cfg: Configs | None = None,
            session: int = 0,
            model: nn.Module = None,
            hist_trainset: MergedDataset = None,
            hist_testset: MergedDataset = None,
            **kwargs
    ) -> None:
        self.sess_cfg = sess_cfg
        self.num_sess = len(sess_cfg.keys())
        self.ncd_cfg = ncd_cfg
        self.kwargs = kwargs

        s_cfg = sess_cfg[f's{session}'].cfg
        cfg.merge_from_config(s_cfg)
        cfg.freeze()

        # Specify historical datasets to store previous data
        if not hist_trainset:
            hist_trainset = MergedDataset()
        self.hist_trainset = hist_trainset

        if not hist_testset:
            hist_testset = MergedDataset()
        self.hist_testset = hist_testset

        super(IncTrainer, self).__init__(
            cfg=cfg,
            session=session,
            model=model,
            max_epochs=cfg.trainer.max_epochs or 1,
            custom_hooks=[NCDHook()],
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
                    cfg=self.cfg,
                    sess_cfg=self.sess_cfg,
                    ncd_cfg=self.ncd_cfg,
                    session=session,
                    model=self.model,
                    hist_trainset=self.hist_trainset,
                    hist_testset=self.hist_testset,
                    **self.kwargs
                )
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__

            super(IncTrainer, self).train()

            # Store historical data
            self.hist_trainset.merge([self.train_loader.dataset], 
                                     replace_transform=True)
            self.hist_testset.merge([self.test_loader.dataset], 
                                    replace_transform=True)
        
        return self.model
