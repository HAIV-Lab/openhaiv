import warnings
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
            buffer: dict | None = None,
            **kwargs
    ) -> None:
        self.sess_cfg = sess_cfg
        self.num_sess = len(sess_cfg.keys())
        self.ncd_cfg = ncd_cfg
        self.kwargs = kwargs

        s_cfg = sess_cfg[f's{session}'].cfg
        cfg.merge_from_config(s_cfg)
        cfg.freeze()

        # Building buffers for inter-session communication
        if buffer is None:
            self.buffer = {}
        else:
            self.buffer = buffer

        # Specify historical datasets to store previous data
        if not hist_trainset:
            hist_trainset = MergedDataset()
        self.hist_trainset = hist_trainset
        self.buffer['hist_trainset'] = hist_trainset

        if not hist_valset:
            hist_valset = MergedDataset()
        self.hist_valset = hist_valset
        self.buffer['hist_valset'] = hist_valset

        if not hist_testset:
            hist_testset = MergedDataset()
        self.hist_testset = hist_testset
        self.buffer['hist_testset'] = hist_testset

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
                    buffer=self.buffer,
                    **self.kwargs
                )
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__

            super(IncTrainer, self).train()
        
        return self.model

    def update_buffer(self, key: str, value, func=None) -> None:
        """Update buffer for inter-session communication.

        Args:
            key (str): Key in buffer.
            value: Value to be stored.
            func (callable, optional): Function to operate on old and new values.
                It is a function of two arguments (old_value, new_value).
        
        Examples:
            >>> update_buffer('key', 1)
            >>> update_buffer('key', 2, func=lambda x, y: x + y)
        """
        if key in self.buffer and func:
            self.buffer[key] = func(self.buffer[key], value)
        else:
            self.buffer[key] = value
 
    def update_hist_trainset(self, new_dataset, replace_transform=False, inplace=False):
        """ Update historical training dataset.

        Args:
            new_dataset (Dataset): New dataset to be merged.
            replace_transform (bool, optional): Replace transform or not. Default: False.
            inplace (bool, optional): If True, use new_dataset to replace hist_trainset.
        """
        # TODO:
        warnings.warn("self.hist_trainset will be deprecated in the future. Use self.buffer['hist_trainset'] instead.")

        if inplace:
            self.hist_trainset = MergedDataset([new_dataset], replace_transform)
        else:
            self.hist_trainset.merge([new_dataset], replace_transform)

        self.buffer['hist_trainset'] = self.hist_trainset

    def update_hist_valset(self, new_dataset, replace_transform=False, inplace=False):
        """ Update historical validation dataset.

        Args:
            new_dataset (Dataset): New dataset to be merged.
            replace_transform (bool, optional): Replace transform or not. Default: False.
            inplace (bool, optional): If True, use new_dataset to replace hist_valset.
        """
        # TODO:
        warnings.warn("self.hist_valset will be deprecated in the future. Use self.buffer['hist_valset'] instead.")

        if inplace:
            self.hist_valset = MergedDataset([new_dataset], replace_transform)
        else:
            self.hist_valset.merge([new_dataset], replace_transform)

        self.buffer['hist_valset'] = self.hist_valset

    def update_hist_testset(self, new_dataset, replace_transform=False, inplace=False):
        """ Update historical testing dataset.

        Args:
            new_dataset (Dataset): New dataset to be merged.
            replace_transform (bool, optional): Replace transform or not. Default: False.
            inplace (bool, optional): If True, use new_dataset to replace hist_testset.
        """
        # TODO:
        warnings.warn("self.hist_testset will be deprecated in the future. Use self.buffer['hist_testset'] instead.")

        if inplace:
            self.hist_testset = MergedDataset([new_dataset], replace_transform)
        else:
            self.hist_testset.merge([new_dataset], replace_transform)

        self.buffer['hist_testset'] = self.hist_testset
