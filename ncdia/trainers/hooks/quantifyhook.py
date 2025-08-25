import os
import torch
from tqdm import tqdm

from ncdia.utils import HOOKS
from .alghook import AlgHook


@HOOKS.register
class QuantifyHook(AlgHook):
    """A hook to quantify and save the statistics of training, evaluation and testing.

    Args:
        gather_train_stats (bool, optional): Whether to gather the statistics of training data. Defaults to False.
        gather_test_stats (bool, optional): Whether to gather the statistics of testing data. Defaults to False.
        save_stats (bool, optional): Whether to save the statistics. Defaults to False.
        verbose (bool, optional): Whether to print the progress. Defaults to False.
    """

    priority = "NORMAL"

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

    def gather_stats(self, model, dataloader, device, id_acc=False, verbose=False) -> dict:
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
        local_features, local_logits = [], []
        local_feats = None
        local_preds = None
        for batch in tbar:
            data = batch["data"].to(device)
            label = batch["label"].to(device)
            if hasattr(model, "evaluate") and callable(getattr(model, "evaluate")):
                preds = model.evaluate(data)
            else:
                preds = model(data)
            # preds = preds[:, :num_classes]
            feats = model.get_features(data)
            if isinstance(feats, tuple) and len(feats) == 2:
                feats, local_feats = feats  # 解包元组
                preds, local_preds = preds
            if isinstance(feats, tuple) and len(feats) == 3:
                feats = feats[1]
                preds = preds[1]
            features.append(feats.clone().detach().cpu())
            logits.append(preds.clone().detach().cpu())
            labels.append(label.clone().detach().cpu())
            if local_preds is not None:
                local_features.append(local_feats.clone().detach().cpu())
                local_logits.append(local_preds.clone().detach().cpu())

        features = torch.cat(features, dim=0)
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0).to(torch.int)
        if local_preds is not None:
            local_features = torch.cat(local_features, dim=0)
            local_logits = torch.cat(local_logits, dim=0)

        # Optionally calculate ID accuracies (top1/top5)
        if id_acc:
            _, preds = logits.topk(5, dim=1)  # Get top-5 predictions
            correct = preds.eq(labels.view(-1, 1).expand_as(preds))

            # Top-1 accuracy
            top1_correct = correct[:, 0].sum().item()
            top1_acc = top1_correct / labels.size(0) * 100

            # Top-5 accuracy
            top5_correct = correct.any(dim=1).sum().item()
            top5_acc = top5_correct / labels.size(0) * 100

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
            "local_features": local_features if local_preds is not None else None,
            "local_logits": local_logits if local_logits is not None else None,
            "top1_acc": top1_acc if id_acc else None,
            "top5_acc": top5_acc if id_acc else None,
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
            verbose=self.verbose,
        )

        # Assign training stats to trainer,
        # which can be accessed in other hooks
        trainer._train_stats = train_stats

        if self.save_stats:
            torch.save(
                train_stats, os.path.join(trainer.work_dir, "train_stats_final.pt")
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
            verbose=self.verbose,
        )

        # Assign testing statistics to trainer,
        # which can be accessed in other hooks
        trainer._test_stats = test_stats

        if self.save_stats:
            torch.save(test_stats, os.path.join(trainer.work_dir, "test_stats.pt"))


@HOOKS.register
class QuantifyHook_OOD(AlgHook):
    """A hook to quantify and save the statistics of training, evaluation and testing.

    Args:
        gather_train_stats (bool, optional): Whether to gather the statistics of training data. Defaults to False.
        gather_test_stats (bool, optional): Whether to gather the statistics of testing data. Defaults to False.
        save_stats (bool, optional): Whether to save the statistics. Defaults to False.
        verbose (bool, optional): Whether to print the progress. Defaults to False.
    """

    priority = "NORMAL"

    def __init__(
        self,
        gather_train_stats=False,
        gather_test_stats=False,
        save_stats=False,
        verbose=False,
    ) -> None:
        super(QuantifyHook_OOD, self).__init__()
        self.gather_train_stats = gather_train_stats
        self.gather_test_stats = gather_test_stats
        self.save_stats = save_stats
        self.verbose = verbose

    def gather_stats(
        self, model, dataloader, device, id_acc=False, verbose=False
    ) -> dict:
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
        local_features, local_logits = [], []
        local_feats = None
        local_preds = None
        for batch in tbar:
            data = batch["data"].to(device)
            label = batch["label"].to(device)
            if hasattr(model, "evaluate") and callable(getattr(model, "evaluate")):
                preds = model.evaluate(data)
            else:
                preds = model(data)
            feats = model.get_features(data)
            if isinstance(preds, tuple) and len(preds) == 2:
                feats, local_feats = feats  # 解包元组
                preds, local_preds = preds
            if isinstance(preds, tuple) and len(preds) == 3:
                feats = feats[1]
                preds = preds[1]
            features.append(feats.clone().detach().cpu())
            logits.append(preds.clone().detach().cpu())
            labels.append(label.clone().detach().cpu())
            if local_preds is not None:
                local_features.append(local_feats.clone().detach().cpu())
                local_logits.append(local_preds.clone().detach().cpu())

        features = torch.cat(features, dim=0)
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0).to(torch.int)
        if local_preds is not None:
            local_features = torch.cat(local_features, dim=0)
            local_logits = torch.cat(local_logits, dim=0)

        if id_acc:
            _, preds = logits.topk(5, dim=1)  # Get top-5 predictions
            correct = preds.eq(labels.view(-1, 1).expand_as(preds))

            # Top-1 accuracy
            top1_correct = correct[:, 0].sum().item()
            top1_acc = top1_correct / labels.size(0) * 100

            # Top-5 accuracy
            top5_correct = correct.any(dim=1).sum().item()
            top5_acc = top5_correct / labels.size(0) * 100

        # Calculate the prototypes
        prototypes = []
        s_prototypes = []
        for cls in torch.unique(labels):
            cls_idx = torch.where(labels == cls)
            s_logits = torch.softmax(logits / 5, dim=1)
            cls_preds = logits[cls_idx]
            prototypes.append(torch.mean(cls_preds, dim=0))
            cls_preds = s_logits[cls_idx]
            s_prototypes.append(torch.mean(cls_preds, dim=0))
        prototypes = torch.stack(prototypes, dim=0)
        s_prototypes = torch.stack(s_prototypes, dim=0)

        return {
            "features": features,
            "logits": logits,
            "labels": labels,
            "prototypes": prototypes,
            "s_prototypes": s_prototypes,
            "local_features": local_features if local_features is not None else None,
            "local_logits": local_logits if local_logits is not None else None,
            "top1_acc": top1_acc if id_acc else None,
            "top5_acc": top5_acc if id_acc else None,
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
            verbose=self.verbose,
        )

        # Assign training stats to trainer,
        # which can be accessed in other hooks
        trainer._train_stats = train_stats

        if self.save_stats:
            torch.save(
                train_stats, os.path.join(trainer.work_dir, "train_stats_final.pt")
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
            verbose=self.verbose,
        )

        # Assign testing statistics to trainer,
        # which can be accessed in other hooks
        trainer._test_stats = test_stats

        if self.save_stats:
            torch.save(test_stats, os.path.join(trainer.work_dir, "test_stats.pt"))
