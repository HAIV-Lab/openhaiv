import torch
import torch.nn as nn

from ncdia.utils import ALGORITHMS
from ncdia.utils.metrics import accuracy
from ncdia.algorithms.base import BaseAlg
from ncdia.utils import HOOKS
from ncdia.trainers.hooks import AlgHook
import numpy as np
from tqdm import tqdm
from .metrics import ood_metrics, search_threshold
import numpy as np
from tqdm import tqdm
from .metrics import ood_metrics, search_threshold
@HOOKS.register
class DMLHook(AlgHook):
    def __init__(self) -> None:
        super().__init__()

    def before_train(self, trainer) -> None:
        trainer.center_optimizer = torch.optim.SGD(trainer.criterion.parameters(), lr=0.5)

    def before_train_iter(self, trainer, batch_idx, data_batch) -> None:
        trainer.center_optimizer.zero_grad()
        epoch = trainer.epoch
        center_milestones = [0, 60, 80]
        assigned_center_weights = [0.0, 0.001, 0.005]
        center_weight = assigned_center_weights[0]
        for i, ms in enumerate(center_milestones):
            if epoch >= ms:
                center_weight = assigned_center_weights[i]
        trainer.center_weight = center_weight

    def after_train_iter(self, trainer, batch_idx, data_batch, outputs) -> None:
        center_weight = trainer.center_weight
        for param in trainer.criterion.parameters():
            param.grad.data *= (1./(center_weight + 1e-12))
        trainer.center_optimizer.step()


@ALGORITHMS.register
class DML(BaseAlg):
    """Decoupling MaxLogit.

    Containing:
        - train_step(trainer, data, label, *args, **kwargs)
        - val_step(trainer, data, label, *args, **kwargs)
        - test_step(trainer, data, label, *args, **kwargs)
        - eval

    """
    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.trainer = trainer

        if trainer.model.loss == 'center':
            hook = DMLHook()
            trainer.register_hook(hook)

    def train_step(self, trainer, data, label, *args, **kwargs):
        """Training step for Decoupling MaxLogit.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Training results. Contains the following keys:
                - "loss": Loss value.
                - "acc": Accuracy value.
        """
        model = trainer.model
        criterion = trainer.criterion
        device = trainer.device

        data, label = data.to(device), label.to(device)
        outputs = model(data)
        if model.loss == 'center':
            features = model.get_features()

            loss_ct = criterion(features, label)
            loss_func = nn.CrossEntropyLoss()
            loss_ce = loss_func(outputs, label)

            loss = loss_ce + loss_ct * trainer.center_weight
        else:
            loss = criterion(outputs, label)
        
        acc = accuracy(outputs, label)[0]

        loss.backward()
        return {"loss": loss.item(), "acc": acc.item()}

    def val_step(self, trainer, data, label, *args, **kwargs):
        """Validation step for Decoupling MaxLogit.

        Args:
            trainer (object): Trainer object.
            data (torch.Tensor): Input data.
            label (torch.Tensor): Label data.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.

        Returns:
            results (dict): Validation results. Contains the following:
                - "loss": Loss value.
                - "acc": Accuracy value.
        """
        model = trainer.model
        device = trainer.device

        data, label = data.to(device), label.to(device)
        outputs = model(data)

        criterion = nn.CrossEntropyLoss()

        loss = criterion(outputs, label)
        acc = accuracy(outputs, label)[0]

        return {"loss": loss.item(), "acc": acc.item()}

    def test_step(self, trainer, data, label, *args, **kwargs):
        """Test step for Decoupling MaxLogit.

        Args:
            id_logits (torch.Tensor): ID logits. Shape (N, C).
            id_feat (torch.Tensor): ID features. Shape (N, D).
            ood_logits (torch.Tensor): OOD logits. Shape (M, C).
            ood_feat (torch.Tensor): OOD features. Shape (M, D).
            train_logits (torch.Tensor): Training logits. Shape (K, C).
            train_feat (torch.Tensor): Training features. Shape (K, D).
            tpr_th (float): True positive rate threshold to compute
                false positive rate. Default is 0.95.
            prec_th (float | None): Precision threshold for searching threshold.
                If None, not searching for threshold. Default is None.

        Returns:
            results (dict): Test results. Contains the following:
                - "loss": Loss value.
                - "acc": Accuracy value.
        """
        return self.val_step(trainer, data, label, *args, **kwargs)
    
    @staticmethod
    def eval(id_gt: torch.Tensor ,id_logits: torch.Tensor, id_feat: torch.Tensor, 
            ood_logits: torch.Tensor, ood_feat: torch.Tensor, 
            train_logits: torch.Tensor = None, train_feat: torch.Tensor = None, 
            tpr_th: float = 0.95, prec_th: float = None, hyparameters: float = None):
        """Decoupled MaxLogit+ (DML+) method for OOD detection.

        Decoupling MaxLogit for Out-of-Distribution Detection
        https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Decoupling_MaxLogit_for_Out-of-Distribution_Detection_CVPR_2023_paper

        Args:
            id_logits (torch.Tensor): ID logits. Shape (N, C).
            id_feat (torch.Tensor): ID features. Shape (N, D).
            ood_logits (torch.Tensor): OOD logits. Shape (M, C).
            ood_feat (torch.Tensor): OOD features. Shape (M, D).
            train_logits (torch.Tensor): Training logits. Shape (K, C).
            train_feat (torch.Tensor): Training features. Shape (K, D).
            tpr_th (float): True positive rate threshold to compute
                false positive rate. Default is 0.95.
            prec_th (float | None): Precision threshold for searching threshold.
                If None, not searching for threshold. Default is None.

        Returns:
            fpr (float): False positive rate.
            auroc (float): Area under the ROC curve.
            aupr_in (float): Area under the precision-recall curve 
                for in-distribution samples.
            aupr_out (float): Area under the precision-recall curve
                for out-of-distribution
        """
        in_score1 = np.max(id_logits.data.cpu().numpy(), axis=1)
        out_score1 = np.max(id_logits.data.cpu().numpy(), axis=1)

        tmp1 = np.sum(in_score1)
        in_score1_tmp = in_score1/tmp1
        out_score1_tmp = out_score1/tmp1

        in_score2 = id_feat.norm(2, dim=1).data.cpu().numpy()
        out_score2 = id_feat.norm(2, dim=1).data.cpu().numpy()

        tmp1 = np.sum(in_score2)
        in_score2_tmp = in_score2/tmp1
        out_score2_tmp = out_score2/tmp1

        in_score = in_score1_tmp + in_score2_tmp  
        out_score = out_score1_tmp + out_score2_tmp

        conf = np.concatenate([in_score, out_score])
        ood_gt = -1 * np.ones(ood_gt)
        label = np.concatenate([id_gt.cpu(), ood_gt])
        if prec_th is None:
            return ood_metrics(conf, label, tpr_th), None
        else:
            return ood_metrics(conf, label, tpr_th), search_threshold(conf, label, prec_th)
