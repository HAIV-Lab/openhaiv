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
        hist_trainset (MergedDataset, optional): Historical training dataset.
        hist_valset (MergedDataset, optional): Historical validation dataset.
        hist_testset (MergedDataset, optional): Historical testing dataset.

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
            hist_valset: MergedDataset = None,
            hist_testset: MergedDataset = None,
            old_model: nn.Module = None,
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

        if not hist_valset:
            hist_valset = MergedDataset()
        self.hist_valset = hist_valset

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
        self.old_model = old_model

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
                    hist_valset=self.hist_valset,
                    hist_testset=self.hist_testset,
                    old_model = self.old_model,
                    **self.kwargs
                )
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__

            super(IncTrainer, self).train()
        
        return self.model

    def update_hist_trainset(self, new_dataset, replace_transform=False, inplace=False):
        """ Update historical training dataset.

        Args:
            new_dataset (Dataset): New dataset to be merged.
            replace_transform (bool, optional): Replace transform or not. Default: False.
            inplace (bool, optional): If True, use new_dataset to replace hist_trainset.
        """
        if inplace:
            self.hist_trainset = MergedDataset([new_dataset], replace_transform)
        else:
            self.hist_trainset.merge([new_dataset], replace_transform)

    def update_hist_valset(self, new_dataset, replace_transform=False, inplace=False):
        """ Update historical validation dataset.

        Args:
            new_dataset (Dataset): New dataset to be merged.
            replace_transform (bool, optional): Replace transform or not. Default: False.
            inplace (bool, optional): If True, use new_dataset to replace hist_valset.
        """
        if inplace:
            self.hist_valset = MergedDataset([new_dataset], replace_transform)
        else:
            self.hist_valset.merge([new_dataset], replace_transform)

    def update_hist_testset(self, new_dataset, replace_transform=False, inplace=False):
        """ Update historical testing dataset.

        Args:
            new_dataset (Dataset): New dataset to be merged.
            replace_transform (bool, optional): Replace transform or not. Default: False.
            inplace (bool, optional): If True, use new_dataset to replace hist_testset.
        """
        if inplace:
            self.hist_testset = MergedDataset([new_dataset], replace_transform)
        else:
            self.hist_testset.merge([new_dataset], replace_transform)
