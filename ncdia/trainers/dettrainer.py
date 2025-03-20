import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ncdia.utils import TRAINERS
from ncdia.dataloader import build_dataloader
from ncdia.algorithms.ood import AutoOOD
from .pretrainer import PreTrainer
from .hooks import QuantifyHook


@TRAINERS.register
class DetTrainer(PreTrainer):
    """Pipeline for OOD detection, 
       including model training and out-of-distribution data detection.

    Args:
        cfg (dict, optional): Configuration for trainer.
        model (nn.Module): Model to be trained.
        eval_loader (DataLoader, optional): DataLoader for OOD evaluation.
        max_epochs (int, optional): Maximum number of training epochs.
        verbose (bool, optional): Whether to print training information.

    Attributes:
        eval_loader (DataLoader): DataLoader for OOD evaluation.
        quantify_hook (QuantifyHook): Hook for dataset statistics collection.
        max_epochs (int): Maximum number of training epochs.
        verbose (bool): Whether to print training information.

    """
    def __init__(
            self,
            cfg: dict | None = None,
            model: nn.Module = None,
            eval_loader: DataLoader | dict | None = None,
            max_epochs: int = 1,
            verbose: bool = False,
            **kwargs
    ) -> None:
        self.kwargs = kwargs
        self.verbose = verbose
        self.quantify_hook = QuantifyHook(verbose=verbose)

        self._eval_loader = {}
        if 'evalloader' in self._cfg:
            self._eval_loader.update(dict(self._cfg['evalloader']))
        if isinstance(eval_loader, dict):
            self._eval_loader.update(eval_loader)
        elif isinstance(eval_loader, DataLoader):
            self._eval_loader = eval_loader
        
        super(DetTrainer, self).__init__(
            cfg=cfg,
            model=model,
            max_epochs=max_epochs,
            custom_hooks=[self.quantify_hook],
            **self.kwargs
        )

    @property
    def eval_loader(self) -> DataLoader:
        """DataLoader: DataLoader for OOD evaluation."""
        if not self._eval_loader:
            return None
        if isinstance(self._eval_loader, dict):
            self._eval_loader, self._eval_dataset_kwargs, \
                self._eval_loader_kwargs = build_dataloader(self._eval_loader)
        return self._eval_loader
    
    @property
    def train_stats(self) -> dict:
        """Get training stats, including features, logits, labels, and prototypes."""
        if "_train_stats" not in self.__dict__:
            return None
        return self._train_stats
    
    def train(self):
        """Training and evaluation for out-of-distribution (OOD) detection.
        Firstly, train a model, and then evaluate the model on the OOD dataset.

        Returns:
            model (nn.Module): Trained model.
        """
        super(DetTrainer, self).train()

        # If the configuration of eval_loader is provided,
        # then evaluate the model on the eval_loader.
        # If not, trainer only train the model but not run OOD detection.
        # Furthermore, OOD detection can also be run after training 
        # by calling `trainer.evaluate(evalloader=DataLoader)`.
        if self.eval_loader:
            self.evaluate()

        return self.model

    def evaluate(
            self,
            metrics: list = ['msp'],
            evalloader: DataLoader = None,
            tpr_th: float = 0.95,
            prec_th: float = None
    ) -> dict:
        """Evaluate dataset.

        Args:
            metrics (list, optional): list of OOD detection methods to evaluate.
            evalloader (DataLoader, optional): DataLoader for OOD evaluation.
            tpr_th (float, optional): True positive rate threshold. Defaults to 0.95.
            prec_th (float, optional): Precision threshold. Defaults to None.

        Returns:
            dict: OOD scores, keys are the names of the OOD detection methods,
                values are the OOD scores and search threshold.
        """
        train_stats = self.train_stats

        eval_stats = self.quantify_hook.gather_stats(
            model=self.model,
            dataloader=evalloader if evalloader else self.eval_loader,
            device=self.device,
            verbose=self.verbose
        )

        scores = AutoOOD().eval(
            metrics=metrics,
            prototype_cls=train_stats['prototypes'],
            fc_weight=self.model.fc.weight.clone().detach().cpu(),
            train_feats=train_stats['features'],
            train_logits=train_stats['logits'],
            id_feats=train_stats['features'],
            id_logits=train_stats['logits'],
            id_labels=train_stats['labels'],
            ood_feats=eval_stats['features'],
            ood_logits=eval_stats['logits'],
            ood_labels=eval_stats['labels'],
            tpr_th=tpr_th,
            prec_th=prec_th
        )

        return scores
        
    def detect(
            self,
            metrics: list = ['msp'],
            evalloader: DataLoader = None,
    ):
        """Detect OOD data, and return evaluated confidence.

        Args:
            metrics (list, optional): list of OOD detection methods to evaluate.
            evalloader (DataLoader, optional): DataLoader for OOD evaluation.

        Returns:
            dict: OOD confidence, keys are the names of the OOD detection methods,
                values are the OOD confidence.
        """
        train_stats = self.train_stats

        eval_stats = self.quantify_hook.gather_stats(
            model=self.model,
            dataloader=evalloader if evalloader else self.eval_loader,
            device=self.device,
            verbose=self.verbose
        )

        confidence = AutoOOD().inference(
            metrics=metrics,
            logits=eval_stats['logits'],
            feat=eval_stats['features'],
            train_logits=train_stats['logits'],
            train_feat=train_stats['features'],
            fc_weight=self.model.fc.weight.clone().detach().cpu(),
            prototype=train_stats['prototypes']
        )

        return confidence
