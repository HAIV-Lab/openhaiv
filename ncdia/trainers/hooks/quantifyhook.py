import os
import torch
from tqdm import tqdm

from ncdia.utils import HOOKS
from .hook import Hook


@HOOKS.register
class QuantifyHook(Hook):
    """A hook to quantify and save the statistics of training, evaluation and testing.

    Args:
        save_stats (bool, optional): Whether to save the statistics. Defaults to True.
        verbose (bool, optional): Whether to print the progress. Defaults to False.
    """

    priority = 'NORMAL'

    def __init__(
            self,
            save_stats=True,
            verbose=False,
    ) -> None:
        super(QuantifyHook, self).__init__()
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
        train_stats = self.gather_stats(
            model=trainer.model,
            dataloader=trainer.train_loader,
            device=trainer.device,
            verbose=self.verbose
        )
        torch.save(train_stats,
                   os.path.join(trainer.work_dir, "train_stats_final.pt"))
