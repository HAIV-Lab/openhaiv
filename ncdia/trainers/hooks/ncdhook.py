from torch.utils.data import DataLoader

from ncdia.utils import HOOKS
from .hook import Hook
from ncdia.algorithms.ncd import AutoNCD


@HOOKS.register
class NCDHook(Hook):
    """A hook to execute OOD and NCD detection to relabel data
    """

    priority = 'NORMAL'

    default_loader = {
        'batch_size': 64,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
    }

    def init_trainer(self, trainer) -> None:
        """Initialize model for trainer.

        Args:
            trainer (BaseTrainer): Trainer object.
        """
        ncd_cfg = trainer.ncd_cfg
        if not ncd_cfg:
            return
        
        loader_cfg = self.default_loader
        loader_cfg.update(ncd_cfg.dataloader or {})
        ncd_detector = AutoNCD(
            trainer.model,
            DataLoader(trainer.hist_trainset, **loader_cfg),
            DataLoader(trainer.hist_testset, **loader_cfg),
            self.device,
            verbose=True,
        )

        trainer.train_loader = ncd_detector.relabel(
            trainer.train_loader,
            metrics=ncd_cfg.metrics or ['msp'],
            tpr_th=ncd_cfg.tpr_th or 0.95,
            prec_th=ncd_cfg.prec_th or 0.42,
        )
