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

    def train(self) -> nn.Module:
        """Incremental training.
        `self.num_sess` determines the number of sessions,
        and session number is stored in `self.session`.

        Returns:
            model (nn.Module): Trained model.
        """
        for session in range(self.num_sess):
            sess_cfg = self.sess_cfg[f's{session}']
            _dset_cfg = sess_cfg.dataset
            if isinstance(_dset_cfg, str):
                _dset_cfg = [_dset_cfg]

            self._session = session
            self._max_epochs = sess_cfg.max_epochs or 1

            dset_cfg = Configs()
            for cfg_file in _dset_cfg:
                dset_cfg.merge_from_yaml(cfg_file)
            dset_cfg.freeze()

            if 'trainloader' in dset_cfg:
                self._train_loader = dict(dset_cfg['trainloader'])
            if 'valloader' in dset_cfg:
                self._val_loader = dict(dset_cfg['valloader'])
            if 'testloader' in dset_cfg:
                self._test_loader = dict(dset_cfg['testloader'])
                
            super(IncTrainer, self).train()
        
        return self.model
