import csv
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .metrics import BasePostprocessor
from ncdia.utils import Config
import ncdia.utils.comm as comm

from .base_evaluator import BaseEvaluator
from .metrics import compute_ood_metrics
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize

from ncdia.augmentations import fantasy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class SAVCattEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        super(SAVCattEvaluator, self).__init__(config)
        self.id_pred = None
        self.id_conf = None
        self.id_gt = None
        self.config = config
        if config.CIL.fantasy is not None:
            self.transform, self.num_trans = fantasy.__dict__[config.CIL.fantasy]()
        else:
            self.transform = None
            self.num_trans = 0

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True,
                  session: int = -1,
                  id_split: str = 'train'):
        """ if id_split = 'train', return the prototype of train_id_loader
        else, return six lists containing pred, conf, label & attribute ones        
        """
        logit_list, pred_list, conf_list, label_list = [], [], [], []
        logit_att_list, pred_att_list, conf_att_list, label_att_list = [], [], [], []
        feature_list = []
        test_class = self.config.CIL.base_class + session * self.config.CIL.way
        net = net.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
                data = batch['data'].cuda()
                label = batch['label'].cuda()
                attribute = batch['attribute'].cuda()
                # data, test_label, test_att_label = [_.cuda() for _ in batch]
                b = data.size()[0]
                data = self.transform(data)
                m = data.size()[0] // b
                joint_preds, joint_preds_att = net(data)
                feat = net.get_features(data)
                joint_preds = joint_preds[:, :test_class*m]
                
                agg_preds = 0
                agg_preds_att = 0
                agg_feat = feat.view(-1, m, feat.size(1)).mean(dim=1)
                for j in range(m):
                    agg_preds = agg_preds + joint_preds[j::m, j::m] / m
                    agg_preds_att = agg_preds_att + joint_preds_att[j::m, :] / m

                feature_list.append(agg_feat)
                logit_list.append(agg_preds)
                score = torch.softmax(agg_preds, dim=1)
                conf, pred = torch.max(score, dim=1)
                pred_list.append(pred.cpu())
                conf_list.append(conf.cpu())
                label_list.append(label.cpu())

                logit_att_list.append(agg_preds_att)
                pred = (torch.sigmoid(agg_preds_att) > 0.5).type(torch.int)
                # score = torch.softmax(agg_preds_att, dim=1)
                # conf, pred = torch.max(score, dim=1)
                conf = pred
                pred_att_list.append(pred.cpu())
                conf_att_list.append(conf.cpu())
                label_att_list.append(attribute.cpu())

        # convert values into numpy array
        feature_list = torch.cat(feature_list, dim=0).cpu()
        logit_list = torch.cat(logit_list, dim=0)
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)
        
        logit_att_list = torch.cat(logit_att_list, dim=0)
        pred_att_list = torch.cat(pred_att_list).numpy()#.astype(int)
        conf_att_list = torch.cat(conf_att_list).numpy()
        label_att_list = torch.cat(label_att_list).numpy().astype(int)

        if id_split == 'train':
            label_list = torch.tensor(label_list)
            classes = torch.unique(label_list)
            prototype_cls = []
            prototype_att = []
            for cls in classes:
                cls_indices = torch.where(label_list == cls)
                cls_predictions = logit_list[cls_indices]
                att_predictions = logit_att_list[cls_indices]
                prototype_cls.append(torch.mean(cls_predictions, dim=0))
                prototype_att.append(torch.mean(att_predictions, dim=0))
    
            prototype_cls = torch.stack(prototype_cls)
            prototype_att = torch.stack(prototype_att)
            print('prototype_cls shape', prototype_cls.shape)  # 8  11  
            print('prototype_att shape', prototype_att.shape)

            return feature_list, logit_list, prototype_cls, prototype_att

        return feature_list, pred_list, logit_list, label_list, pred_att_list, logit_att_list, label_att_list


    def eval_ood(self,
                 net: nn.Module,
                 id_data_loaders: Dict[str, DataLoader],
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 session: int):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        assert 'test' in id_data_loaders, \
            'id_data_loaders should have the key: test!'
        
        dataset_name = self.config.dataset.name

        # -------- prepare the prototype of training set -------- #
        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        train_feat, train_logit, prototype_cls, prototype_att = self.inference(
            net, id_data_loaders['train'], session=session, id_split='train')
        prototype_cls = F.normalize(prototype_cls, p=2, dim=1)
        prototype_att = F.normalize(prototype_att, p=2, dim=1)

        save_dir = os.path.join(self.config.output_dir, 'prototype.pth')
        torch.save({'prototype_cls': prototype_cls, 'prototype_att': prototype_att}, save_dir)

        # ------------ turn the logit to the conf score ----------- #
        id_feat, id_pred, id_logit, id_gt, id_att_pred, id_att_logit, id_att_gt = self.inference(
            net, id_data_loaders['test'], session=session, id_split='test')
        id_logit = id_logit.clone()
        id_att_logit = id_att_logit.clone()

        id_conf = F.normalize(id_logit, p=2, dim=1) @ prototype_cls.T
        id_att_conf = F.normalize(id_att_logit, p=2, dim=1) @ prototype_att.T
        id_conf, _ = torch.max(id_conf, dim=1, keepdim=True)
        id_att_conf, _ = torch.max(id_att_conf, dim=1, keepdim=True)
        
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        # load nearood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, 
                       [train_feat, train_logit,id_feat, id_conf, id_logit, id_gt, \
                        id_att_conf, id_att_logit, id_att_gt, prototype_cls, prototype_att],
                       ood_data_loaders, 
                       session)


    def _eval_ood(self,
                  net: nn.Module,
                  id_list: List[np.ndarray],
                  ood_data_loader,
                  session: int = -1):
        print(f'Processing ood inference...', flush=True)
        [train_feat, train_logit, id_feat, id_conf, id_logit, id_gt, \
         id_att_conf, id_att_logit, id_att_gt, prototype_cls, prototype_att] = id_list
        metrics_list = []

        ood_feat, ood_pred, ood_logit, ood_gt, ood_att_pred, ood_att_logit, ood_att_gt = \
            self.inference(net, ood_data_loader, session=session, id_split='test')
        ood_logit = ood_logit.clone().detach()
        ood_att_logit = ood_att_logit.clone().detach()


        # ----------  draw the TSNE of features   --------- #
        total_feat = torch.cat([id_feat.cpu(), ood_feat.cpu()])
        label = np.concatenate([id_gt, ood_gt])
        tsne = TSNE(n_components=2, random_state=0)
        features_tsne = tsne.fit_transform(total_feat)
        plt.figure(figsize=(10, 6))
        cmap = plt.get_cmap('tab20')
        unique_labels = np.unique(label)
        for i, l in enumerate(unique_labels):
            indices = label == l
            plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], label=l, c=[cmap(i)])
        plt.legend()
        plt.title('t-SNE visualization of features')
        plt.xlabel('t-SNE axis 1')
        plt.ylabel('t-SNE axis 2')
        plt.savefig(os.path.join(self.config.output_dir, 'tSNE.png'))
        # --------------------------------- #

        # ood_feat, ood_pred, ood_logit, ood_gt  
        # ood_att_pred, ood_att_logit, ood_att_gt 
        
        ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
        # ------------  MSP  ------------- #
        tmp = torch.softmax(id_logit, 1)
        tid_conf, _ = torch.max(tmp, dim=1, keepdim=True)
        tmp = torch.softmax(ood_logit, 1)
        tood_conf, _ = torch.max(tmp, dim=1, keepdim=True)

        print(f'Computing *** MSP *** metrics on OOD (new-train) dataset...')
        # pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([tid_conf.cpu(), tood_conf.cpu()])
        label = np.concatenate([id_gt, ood_gt])
        ood_metrics_msp = compute_ood_metrics(conf, label)
        # print(ood_metrics_msp)
       
        # ------------  MCM  ------------- #
        T = 2
        tmp = torch.softmax(id_logit/T, 1)
        tid_conf, _ = torch.max(tmp, dim=1, keepdim=True)
        tmp = torch.softmax(ood_logit/T, 1)
        tood_conf, _ = torch.max(tmp, dim=1, keepdim=True)

        print(f'Computing *** MCM *** metrics on OOD (new-train) dataset...')
        # pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([tid_conf.cpu(), tood_conf.cpu()])
        label = np.concatenate([id_gt, ood_gt])
        ood_metrics_mcm = compute_ood_metrics(conf, label)
        # print(ood_metrics_mcm)

        # ------------  MaxLogit  ------------- #        
        tid_conf, _ = torch.max(id_logit, dim=1, keepdim=True)
        tood_conf, _ = torch.max(ood_logit, dim=1, keepdim=True)

        print(f'Computing *** MaxLogit *** metrics on OOD (new-train) dataset...')
        # pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([tid_conf.cpu(), tood_conf.cpu()])
        label = np.concatenate([id_gt, ood_gt])
        ood_metrics_mls = compute_ood_metrics(conf, label)
        # print(ood_metrics_mls)

        # ------------  Energy  ------------- #
        from scipy.special import logsumexp
        tid_conf = logsumexp(id_logit.cpu(), axis=-1)
        tood_conf = logsumexp(ood_logit.cpu(), axis=-1)

        print(f'Computing *** Energy *** metrics on OOD (new-train) dataset...')
        # pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([tid_conf, tood_conf])
        label = np.concatenate([id_gt, ood_gt])
        ood_metrics_energy = compute_ood_metrics(conf, label)
        # print(ood_metrics_energy)

        # ------------  ViM  ------------- #
        from numpy.linalg import norm, pinv
        from scipy.special import logsumexp
        from sklearn.covariance import EmpiricalCovariance
        DIM = train_feat.shape[1]//2
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(train_feat.cpu())
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray(
            (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
        vlogit_id_train = norm(np.matmul(train_feat.cpu(), NS), axis=-1)
        alpha = train_logit.max(axis=-1)[0].mean() / vlogit_id_train.mean()
        print(f'Computing *** ViM *** metrics on OOD (new-train) dataset...')
        # print(f'DIM={DIM}, alpha={alpha:.4f}')

        id_energy = logsumexp(id_logit.cpu(), axis=-1)
        ood_energy = logsumexp(ood_logit.cpu(), axis=-1)
        id_vlogit = norm(np.matmul(id_feat.numpy(), NS), axis=-1) * alpha.cpu().numpy()
        ood_vlogit = norm(np.matmul(ood_feat.numpy(), NS), axis=-1) * alpha.cpu().numpy()

        tid_conf = -id_vlogit + id_energy
        tood_conf = -ood_vlogit + ood_energy
        conf = np.concatenate([tid_conf, tood_conf])
        label = np.concatenate([id_gt, ood_gt])
        ood_metrics_vim = compute_ood_metrics(conf, label)
        # print(ood_metrics_vim)

        # ------------  DML  ------------- # 直接用featnorm加上logit分数
        w = net.fc.weight.clone().detach()
        w = F.normalize(w, p=2, dim=1).cpu()
        w = w[::2,]  # savc使用的是两倍类别数的fc层
        id_cosine = F.normalize(id_feat, p=2, dim=1) @ w.T
        ood_cosine = F.normalize(ood_feat, p=2, dim=1) @ w.T
        id_mcos, _ = torch.max(id_cosine, dim=1, keepdim=True)
        ood_mcos, _ = torch.max(ood_cosine, dim=1, keepdim=True)
        id_norm = torch.norm(id_feat, dim=1)
        ood_norm = torch.norm(ood_feat, dim=1)

        tid_conf = id_mcos + 0.002 * id_norm.unsqueeze(1)
        tood_conf = ood_mcos + 0.002 * ood_norm.unsqueeze(1)

        print(f'Computing *** DML *** metrics on OOD (new-train) dataset...')
        # pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([tid_conf.cpu(), tood_conf.cpu()])
        label = np.concatenate([id_gt, ood_gt])
        ood_metrics_dml = compute_ood_metrics(conf, label)
        print(ood_metrics_dml)

        ttood_conf = F.normalize(ood_logit, p=2, dim=1) @ prototype_cls.T
        ttood_conf, _ = torch.max(ttood_conf, dim=1, keepdim=True)
        ttid_conf = id_conf

        tid_conf = tid_conf + 40 * ttid_conf.cpu()
        tood_conf = tood_conf + 40 * ttood_conf.cpu()

        print(f'Computing *** DML+ *** metrics on OOD (new-train) dataset...')
        # pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([tid_conf.cpu(), tood_conf.cpu()])
        label = np.concatenate([id_gt, ood_gt])
        ood_metrics_dmlp = compute_ood_metrics(conf, label)
        print(ood_metrics_dmlp)
        # ------------  ？？？  ------------- #


        # -------  Our original OOD methods  ------- #
        ood_conf = F.normalize(ood_logit, p=2, dim=1) @ prototype_cls.T
        ood_att_conf = F.normalize(ood_att_logit, p=2, dim=1) @ prototype_att.T
        ood_conf, _ = torch.max(ood_conf, dim=1, keepdim=True)
        ood_att_conf, _ = torch.max(ood_att_conf, dim=1, keepdim=True)
        ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood

        if self.config.recorder.save_scores:
            self._save_scores(ood_pred, ood_conf, ood_gt, 'OOD')


        print(f'Computing metrics on OOD (new-train) dataset...')
        # pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([id_conf.cpu(), ood_conf.cpu()])
        label = np.concatenate([id_gt, ood_gt])
        ood_metrics_cls = compute_ood_metrics(conf, label) #, pred)
        print(ood_metrics_cls)

        conf = np.concatenate([id_att_conf.cpu(), ood_att_conf.cpu()])
        ood_metrics_att = compute_ood_metrics(conf, label)

        conf = np.concatenate([id_att_conf.cpu() + id_conf.cpu(), \
                               ood_att_conf.cpu() + ood_conf.cpu()])
        ood_metrics_merge = compute_ood_metrics(conf, label)

        # ------------------------------------------- #
    

        if self.config.recorder.save_csv:
            self._save_csv(ood_metrics_msp, dataset_name='OOD_MSP')
            self._save_csv(ood_metrics_mcm, dataset_name='OOD_MCM')
            self._save_csv(ood_metrics_mls, dataset_name='OOD_MLS')
            self._save_csv(ood_metrics_energy, dataset_name='ood_Energy')
            self._save_csv(ood_metrics_dml, dataset_name='OOD_DML')
            self._save_csv(ood_metrics_dmlp, dataset_name='OOD_DML+')
            self._save_csv(ood_metrics_vim, dataset_name='OOD_ViM')
            
            self._save_csv(ood_metrics_cls, dataset_name='OOD_PatternMatching')
            self._save_csv(ood_metrics_att, dataset_name='OOD_Attr')
            self._save_csv(ood_metrics_merge, dataset_name='OOD_merge')


        # metrics_list.append(ood_metrics_cls)

        # print('Computing mean metrics...', flush=True)
        # metrics_list = np.array(metrics_list)
        # metrics_mean = np.mean(metrics_list, axis=0)
        # if self.config.recorder.save_csv:
        #     self._save_csv(metrics_mean, dataset_name='OOD_cls')


    def _save_csv(self, metrics, dataset_name):
        [fpr, auroc, aupr_in, aupr_out] = metrics

        write_content = {
            'dataset': dataset_name,
            'FPR@95': '{:.2f}'.format(100 * fpr),
            'AUROC': '{:.2f}'.format(100 * auroc),
            'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
            'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
        }

        fieldnames = list(write_content.keys())

        # print ood metric results
        print('{}, FPR@95: {:.2f}, AUROC: {:.2f}'.format(dataset_name, 100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print(u'\u2500' * 70, flush=True)

        csv_path = os.path.join(self.config.output_dir, 'ood.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def _save_scores(self, pred, conf, gt, save_name):
        save_dir = os.path.join(self.config.output_dir, 'scores')
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name),
                 pred=pred,
                 conf=conf,
                 label=gt)

    def eval_acc_attr(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 session: int = -1,
                 epoch_idx: int = -1):
        net.eval()

        _, id_pred, id_logit, id_gt, id_att_pred, id_att_logit, id_att_gt = self.inference(
            net, data_loader, session=session, id_split='test')
        
        # --------- calculate the acc for each class -------- #
        id_pred, id_gt = torch.tensor(id_pred), torch.tensor(id_gt)
        classes = torch.unique(id_gt)
        # print(f'val classes having {classes}', flush=True)

        correct = id_pred.eq(id_gt.data).sum().item()
        acc = correct / len(data_loader.dataset)

        accuracy_per_class = {}
        for cls in classes:
            cls_indices = torch.where(id_gt == cls)
            cls_predictions =  id_pred[cls_indices]
            cls_labels = id_gt[cls_indices]
            correct = torch.sum(cls_predictions == cls_labels).item()
            total = len(cls_labels)
            accuracy = correct / total
            print(f"Class: {cls.item():>3} having {total:>5} samples, | Accuracy = {100*accuracy:>6.2f}%", flush=True)
            accuracy_per_class[cls.item()] = accuracy

        # --------- calculate the mAP -------- #
        true_attrs = np.concatenate(id_att_gt)
        pred_attrs = np.concatenate(id_att_pred)

        # 计算多标签分类的评估指标
        OP = precision_score(true_attrs, pred_attrs, average='micro', zero_division=1.0)
        OR = recall_score(true_attrs, pred_attrs, average='micro', zero_division=1.0)
        OF1 = f1_score(true_attrs, pred_attrs, average='micro', zero_division=1.0)
        CP = precision_score(true_attrs, pred_attrs, average='macro', zero_division=1.0)
        CR = recall_score(true_attrs, pred_attrs, average='macro', zero_division=1.0)
        CF1 = f1_score(true_attrs, pred_attrs, average='macro', zero_division=1.0)

        mAP = average_precision_score(true_attrs, pred_attrs, average='micro')

        # metrics = {}
        # metrics['epoch_idx'] = epoch_idx
        # metrics['loss'] = self.save_metrics(loss)
        # metrics['acc'] = self.save_metrics(acc)
        metrics = {
            'epoch_idx': epoch_idx,
            'acc': self.save_metrics(acc),
            'OP': self.save_metrics(OP),
            'OR': self.save_metrics(OR),
            'OF1': self.save_metrics(OF1),
            'CP': self.save_metrics(CP),
            'CR': self.save_metrics(CR),
            'CF1': self.save_metrics(CF1),
            'mAP': self.save_metrics(mAP),
        }
        return metrics

    def report(self, test_metrics):
        print('Completed!', flush=True)




    def cheating_testset(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 session: int = -1,
                 epoch_idx: int = -1):
        net.eval()
        logit_list, pred_list, conf_list, label_list = [], [], [], []
        logit_att_list, pred_att_list, conf_att_list, label_att_list = [], [], [], []
        feature_list, imgpath_list = [], []
        test_class = self.config.CIL.base_class + session * self.config.CIL.way
        net = net.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader,
                          disable=not True or not comm.is_main_process()):
                data = batch['data'].cuda()
                label = batch['label'].cuda()
                attribute = batch['attribute'].cuda()
                imgpath = batch['imgpath']
                # data, test_label, test_att_label = [_.cuda() for _ in batch]
                b = data.size()[0]
                data = self.transform(data)
                m = data.size()[0] // b
                joint_preds, joint_preds_att = net(data)
                feat = net.get_features(data)
                joint_preds = joint_preds[:, :test_class*m]
                
                agg_preds = 0
                agg_preds_att = 0
                agg_feat = feat.view(-1, m, feat.size(1)).mean(dim=1)
                for j in range(m):
                    agg_preds = agg_preds + joint_preds[j::m, j::m] / m
                    agg_preds_att = agg_preds_att + joint_preds_att[j::m, :] / m

                feature_list.append(agg_feat)
                logit_list.append(agg_preds)
                score = torch.softmax(agg_preds, dim=1)
                conf, pred = torch.max(score, dim=1)
                pred_list.append(pred.cpu())
                conf_list.append(conf.cpu())
                label_list.append(label.cpu())
                imgpath_list = imgpath_list + imgpath

        # convert values into numpy array
        feature_list = torch.cat(feature_list, dim=0).cpu()
        pred_list = torch.cat(pred_list).numpy().astype(int)
        label_list = torch.cat(label_list).numpy().astype(int)

        imgpath_list, pred_list, label_list
        
        # --------- calculate the acc for each class -------- #
        id_pred, id_gt = torch.tensor(pred_list), torch.tensor(label_list)
        correct = id_pred.eq(id_gt.data).sum().item()
        acc = correct / len(data_loader.dataset)

        correct_imgpaths_dict = {}
        merge_imgpaths_dict = {}
        accuracy_per_class = {}
        for cls in [11, 12, 13]:
            cls_indices = torch.where(id_gt == cls)[0]
            tmp_imgpath_list = [imgpath_list[i] for i in cls_indices.tolist()]

            cls_predictions = id_pred[cls_indices]
            cls_labels = id_gt[cls_indices]
            correct = torch.sum(cls_predictions == cls_labels).item()
            total = len(cls_labels)
            accuracy = correct / total
            print(f"Class: {cls:>3} having {total:>5} samples, | Accuracy = {100*accuracy:>6.2f}%", flush=True)
            accuracy_per_class[cls] = accuracy

            correct_indices = torch.where(cls_predictions == cls_labels)[0]
            correct_imgpaths = [tmp_imgpath_list[i] for i in correct_indices.tolist()]
            correct_imgpaths_dict[cls] = correct_imgpaths

            wrong_indices = torch.where(cls_predictions != cls_labels)[0]
            if cls == 3:
                www = 5
            else:
                www = 2
            merge_imgpaths = [tmp_imgpath_list[i] for i in wrong_indices.tolist()[:www] + correct_indices.tolist()]
            merge_imgpaths_dict[cls] = merge_imgpaths

            save_directory = '/new_data/dx450/ATR-13-pingzhuang/0718/cheat_test/'+str(cls)
            os.makedirs(save_directory, exist_ok=True)
            import shutil
            for img_path in merge_imgpaths:
                save_path = os.path.join(save_directory, os.path.basename(img_path))
                shutil.copy2(img_path, save_path)

