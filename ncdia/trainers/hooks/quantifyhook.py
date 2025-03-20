import os
import torch
from tqdm import tqdm

from ncdia.utils import HOOKS
from .alghook import AlgHook


@HOOKS.register
class QuantifyHook(AlgHook):
    """A hook to quantify and save the statistics of training, evaluation and testing.

    Args:
        save_stats (bool, optional): Whether to save the statistics. Defaults to False.
        verbose (bool, optional): Whether to print the progress. Defaults to False.
    """

    priority = 'NORMAL'

    def __init__(
            self,
            gather_train_stats=False,
            gather_test_stats=False,
            save_stats=False,
            verbose=False,
    ) -> None:
        super(QuantifyHook, self).__init__()
        self.gather_train_stats = gather_train_stats
        self.gather_test_stats = gather_test_stats
        self.save_stats = save_stats
        self.verbose = verbose

    def gather_stats(self, model, dataloader, device, verbose=False) -> dict:
        """Calculate the statistics of dataset.

        Args:
            model (nn.Module): The model.
            dataloader (DataLoader): The data loader.
            device (torch.device): The device.
            verbose (bool, optional): Whether to print the progress.

        Returns:
            dict: The statistics of dataset.
        """
        num_classes = dataloader.dataset.num_classes

        if verbose:
            tbar = tqdm(dataloader, dynamic_ncols=True, desc="Stats")
        else:
            tbar = dataloader

        # Gather the statistics
        features, logits, labels = [], [], []
        for batch in tbar:
            data = batch['data'].to(device)
            label = batch['label'].to(device)
            preds = model(data)
            preds = preds[:, :num_classes]
            feats = model.get_features(data)

            features.append(feats.clone().detach().cpu())
            logits.append(preds.clone().detach().cpu())
            labels.append(label.clone().detach().cpu())
        
        features = torch.cat(features, dim=0)
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0).to(torch.int)

        # Calculate the prototypes
        prototypes = []
        for cls in torch.unique(labels):
            cls_idx = torch.where(labels == cls)
            cls_preds = logits[cls_idx]
            prototypes.append(torch.mean(cls_preds, dim=0))
        prototypes = torch.stack(prototypes, dim=0)

        return {
            "features": features,
            "logits": logits,
            "labels": labels,
            "prototypes": prototypes,
        }

    def after_train(self, trainer) -> None:
        """Calculate the statistics of training data.

        Args:
            trainer (Trainer): The trainer.
        """
        if not self.gather_train_stats:
            return

        train_stats = self.gather_stats(
            model=trainer.model,
            dataloader=trainer.train_loader,
            device=trainer.device,
            verbose=self.verbose
        )

        # Assign training stats to trainer,
        # which can be accessed in other hooks
        trainer._train_stats = train_stats

        if self.save_stats:
            torch.save(
                train_stats,
                os.path.join(trainer.work_dir, "train_stats_final.pt")
            )

    def after_test(self, trainer) -> None:
        """Calculate the statistics of testing data.

        Args:
            trainer (Trainer): The trainer
        """
        if not self.gather_test_stats:
            return

        test_stats = self.gather_stats(
            model=trainer.model,
            dataloader=trainer.test_loader,
            device=trainer.device,
            verbose=self.verbose
        )

        # Assign testing statistics to trainer,
        # which can be accessed in other hooks
        trainer._test_stats = test_stats

        if self.save_stats:
            torch.save(
                test_stats,
                os.path.join(trainer.work_dir, "test_stats.pt")
            )
