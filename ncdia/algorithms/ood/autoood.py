import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .methods import (
    msp, mcm, max_logit, energy, vim, dml, dmlp, prot
)

from ncdia.utils.cfg import Configs

class AutoOOD(object):
    """AutoOOD class for evaluating OOD detection methods.

    Args:
        device (torch.device): device to run the evaluation

    """
    def __init__(self, device: torch.device, cfg: Configs):
        super(AutoOOD, self).__init__()
        self.device = device
        self.config = cfg
        self.transform = None

    @torch.no_grad()
    def inference(
            self, 
            model: nn.Module, 
            dataloader: DataLoader,
            session: int,
            split: str = 'train',
    ) -> tuple:
        """Inference the model on the dataloader and return the feature, prediction, logit, and label.
        If split is 'train', return the prototype of the training data.

        Args:
            model (nn.Module): model to be evaluated
            dataloader (DataLoader): dataloader for evaluation
            session (int): session number
            split (str, optional): train or test. Defaults to 'train'.

        Returns:
            If split is 'train':
                tuple: feature_list, logit_list, prototype_cls, prototype_att
            If split is 'test':
                tuple: feature_list, pred_list, logit_list, label_list, pred_att_list, logit_att_list, label_att_list
        """
        model.eval()
        test_class = self.config.CIL.base_class + session * self.config.CIL.way
        logit_list, pred_list, conf_list, label_list = [], [], [], []
        # logit_att_list, pred_att_list, conf_att_list, label_att_list = [], [], [], []
        feature_list = []

        for batch in tqdm(dataloader, dynamic_ncols=True):
            data = batch['data'].to(self.device)
            label = batch['label'].to(self.device)
            # attribute = batch['attribute'].to(self.device)
            
            b = data.size(0)
            if self.transform is not None:
                data = self.transform(data)
            m = data.size(0) // b
            joint_preds = model(data)
            feat = model.get_features(data)
            joint_preds = joint_preds[:, :test_class*m]
            
            agg_preds = 0
            # agg_preds_att = 0
            agg_feat = feat.view(-1, m, feat.size(1)).mean(dim=1)
            for j in range(m):
                agg_preds = agg_preds + joint_preds[j::m, j::m] / m
                # agg_preds_att = agg_preds_att + joint_preds_att[j::m, :] / m

            feature_list.append(agg_feat)
            logit_list.append(agg_preds)
            score = torch.softmax(agg_preds, dim=1)
            conf, pred = torch.max(score, dim=1)
            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

            # logit_att_list.append(agg_preds_att)
            # pred = (torch.sigmoid(agg_preds_att) > 0.5).type(torch.int)
            # conf = pred
            # pred_att_list.append(pred.cpu())
            # conf_att_list.append(conf.cpu())
            # label_att_list.append(attribute.cpu())

        # convert values into numpy array
        feature_list = torch.cat(feature_list, dim=0).cpu()
        logit_list = torch.cat(logit_list, dim=0)
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)
        
        # logit_att_list = torch.cat(logit_att_list, dim=0)
        # pred_att_list = torch.cat(pred_att_list).numpy()
        # conf_att_list = torch.cat(conf_att_list).numpy()
        # label_att_list = torch.cat(label_att_list).numpy().astype(int)

        if split == 'train':
            label_list = torch.tensor(label_list)
            classes = torch.unique(label_list)
            prototype_cls = []

            for cls in classes:
                cls_indices = torch.where(label_list == cls)
                cls_predictions = logit_list[cls_indices]
                # att_predictions = logit_att_list[cls_indices]
                prototype_cls.append(torch.mean(cls_predictions, dim=0))
                # prototype_att.append(torch.mean(att_predictions, dim=0))

            return feature_list, logit_list, torch.stack(prototype_cls)

        return feature_list, pred_list, logit_list, label_list

    def eval(
            self, 
            model: nn.Module, 
            id_trainloader: DataLoader, 
            id_testloader: DataLoader,
            ood_dataloader: DataLoader,
            session: int,
    ) -> dict:
        """Evaluate the OOD detection methods and return OOD scores.

        Args:
            model (nn.Module): model to be evaluated
            id_trainloader (DataLoader): dataloader for training data
            id_testloader (DataLoader): dataloader for test data
            ood_dataloader (DataLoader): dataloader for OOD data
            session (int): session number

        Returns:
            dict: OOD scores. Contains:
                'msp', 'mcm', 'maxlogit', 'energy', 
                'vim', 'dml', 'dmlp', 'cls', 'att', 'merge'
        """
        model.eval()

        # prepare prototype of training data
        train_feat, train_logit, prototype_cls = \
            self.inference(model, id_trainloader, session=session, split='train')
        prototype_cls = F.normalize(prototype_cls, p=2, dim=1)
        # prototype_att = F.normalize(prototype_att, p=2, dim=1)
        
        # prepare id statistics from test data
        id_feat, id_pred, id_logit, id_gt = \
            self.inference(model, id_testloader, session=session, split='test')
        id_logit = id_logit.detach().clone()
        # id_att_logit = id_att_logit.detach().clone()

        # prepare ood statistics from ood data
        ood_feat, ood_pred, ood_logit, ood_gt = \
            self.inference(model, ood_dataloader, session=session, split='test')
        ood_logit = ood_logit.detach().clone()
        # ood_att_logit = ood_att_logit.detach().clone()

        # calculate ood score
        ood_msp = msp(id_gt, id_logit, ood_gt, ood_logit)
        # ood_mcm = mcm(id_gt, id_logit, ood_gt, ood_logit)
        # ood_maxlogit = max_logit(id_gt, id_logit, ood_gt, ood_logit)
        # ood_energy = energy(id_gt, id_logit, ood_gt,ood_logit)
        # ood_vim = vim(id_gt, id_logit, id_feat, ood_gt, ood_logit, ood_feat, train_logit, train_feat)
        # ood_dml = dml(id_gt, id_feat, ood_gt, ood_feat, model.fc.weight)
        # ood_dmlp = dmlp(id_gt, id_logit, id_feat, ood_gt, ood_logit, ood_feat, model.fc.weight, prototype_cls)
        # ood_cls = prot(id_gt, [id_logit], ood_gt, [ood_logit], [prototype_cls])
        # ood_att = prot(id_gt, [id_att_logit], ood_gt, [ood_att_logit], [prototype_att])
        # ood_merge = prot(id_gt, [id_logit, id_att_logit], ood_gt, [ood_logit, ood_att_logit], [prototype_cls, prototype_att])

        return {
            'msp': ood_msp,
            # 'mcm': ood_mcm,
            # 'maxlogit': ood_maxlogit,
            # 'energy': ood_energy,
            # 'vim': ood_vim,
            # 'dml': ood_dml,
            # 'dmlp': ood_dmlp,
            # 'cls': ood_cls,
            # 'merge': ood_merge,
        }
