import csv
import os


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from ncdia.augmentations import fantasy

from ncdia.utils.cfg import Configs


class NCDDiscover(object):
    def __init__(self, config: Configs):
        self.id_pred = None
        self.id_conf = None
        self.id_gt = None
        self.config = config
        # print("config.CIL.fantasy: ", config.CIL.fantasy)
        if config.CIL.fantasy is not None and config.CIL.fantasy != "None":
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
        else, return six lists containing pred, conf, label  ones        
        """
        logit_list, pred_list, conf_list, label_list = [], [], [], []
        imgpath_list, feature_list = [], []
        test_class = self.config.CIL.base_class + session * self.config.CIL.way
        net = net.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
                data = batch['data'].cuda()
                label = batch['label'].cuda()
                imgpath = batch['imgpath']

                b = data.size()[0]
                # 20240712
                if self.transform is not None:
                    data = self.transform(data)
                m = data.size()[0] // b
                joint_preds = net(data)
                feat = net.get_features(data)
                joint_preds = joint_preds[:, :test_class*m]
                
                agg_preds = 0
                agg_feat = feat.view(-1, m, feat.size(1)).mean(dim=1)
                for j in range(m):
                    agg_preds = agg_preds + joint_preds[j::m, j::m] / m

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
        logit_list = torch.cat(logit_list, dim=0)
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).cpu()
        label_list = torch.cat(label_list).numpy().astype(int)

        if id_split == 'train':
            label_list = torch.tensor(label_list)
            classes = torch.unique(label_list)
            prototype_cls = []
            for cls in classes:
                cls_indices = torch.where(label_list == cls)
                cls_predictions = logit_list[cls_indices]
                prototype_cls.append(torch.mean(cls_predictions, dim=0))
    
            prototype_cls = torch.stack(prototype_cls)
            print('prototype_cls shape', prototype_cls.shape)  # 8  11  

            return prototype_cls

        return imgpath_list, feature_list, pred_list, logit_list, label_list

    def get_pseudo_newloader(self,
                 net: nn.Module,
                 id_data_loaders: Dict[str, DataLoader],
                 ood_data_loader,
                 train_transfrom, 
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
        prototype_cls = self.inference(net, id_data_loaders['train'], session=session, id_split='train')
        print('prototype_cls shape', prototype_cls.shape)
        prototype_cls = F.normalize(prototype_cls, p=2, dim=1)

        # ------------ turn the logit to the conf score ----------- #
        id_imgpath_list, id_feat, id_pred, id_logit, id_gt = self.inference(
            net, id_data_loaders['test'], session=session, id_split='test')
        id_logit = id_logit.clone().detach()

        id_conf = F.normalize(id_logit, p=2, dim=1) @ prototype_cls.T
        id_conf, _ = torch.max(id_conf, dim=1, keepdim=True)
        
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        # load nearood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        print(f'Performing inference on OOD dataset...', flush=True)
        ood_imgpath_list, ood_feat, ood_pred, ood_logit, ood_gt= self.inference(
            net, ood_data_loader, session=session, id_split='test')
        ood_logit = ood_logit.clone().detach()

        ood_conf = F.normalize(ood_logit, p=2, dim=1) @ prototype_cls.T
        ood_conf, _ = torch.max(ood_conf, dim=1, keepdim=True)

        # !!!! ood_neg_gt set to -1 as ood, ood_gt is the original one (postive number)
        ood_neg_gt = -1 * np.ones_like(ood_gt) 

        # ------------ OOD detection on "val set" to find the threshold ----------- #
        print(f"Using the OOD info type: {self.config.discoverer.ood_type}")
        id_conf, ood_conf = id_conf.cpu(), ood_conf.cpu()
        # 这个conf是id的大，ood的小
        id_len, ood_len = id_conf.shape[0], ood_conf.shape[0]
        ratio = self.config.discoverer.val_ratio
        if self.config.discoverer.ood_type == 'MLS':
            tid_conf, _ = torch.max(id_logit.cpu(), dim=1, keepdim=True)
            tood_conf, _ = torch.max(ood_logit.cpu(), dim=1, keepdim=True)
            conf = torch.cat([tid_conf[:int(id_len * ratio)], tood_conf[:int(ood_len * ratio)]])
            label = np.concatenate([id_gt[:int(id_len * ratio)], ood_neg_gt[:int(ood_len * ratio)]])
        elif self.config.discoverer.ood_type == 'MSP':
            tmp = torch.softmax(id_logit.cpu(), 1)
            tid_conf, _ = torch.max(tmp, dim=1, keepdim=True)
            tmp = torch.softmax(ood_logit.cpu(), 1)
            tood_conf, _ = torch.max(tmp, dim=1, keepdim=True)
            conf = torch.cat([tid_conf[:int(id_len * ratio)], tood_conf[:int(ood_len * ratio)]])
            label = np.concatenate([id_gt[:int(id_len * ratio)], ood_neg_gt[:int(ood_len * ratio)]])
        elif self.config.discoverer.ood_type == 'MCM':
            T = 2
            tmp = torch.softmax(id_logit.cpu() / T, 1)
            tid_conf, _ = torch.max(tmp, dim=1, keepdim=True)
            tmp = torch.softmax(ood_logit.cpu() / T, 1)
            tood_conf, _ = torch.max(tmp, dim=1, keepdim=True)
            conf = torch.cat([tid_conf[:int(id_len * ratio)], tood_conf[:int(ood_len * ratio)]])
            label = np.concatenate([id_gt[:int(id_len * ratio)], ood_neg_gt[:int(ood_len * ratio)]])
        elif self.config.discoverer.ood_type == 'Energy':
            from scipy.special import logsumexp
            tid_conf = torch.tensor(logsumexp(id_logit.cpu(), axis=-1))
            tood_conf = torch.tensor(logsumexp(ood_logit.cpu(), axis=-1))
            conf = torch.cat([tid_conf[:int(id_len * ratio)], tood_conf[:int(ood_len * ratio)]])
            label = np.concatenate([id_gt[:int(id_len * ratio)], ood_neg_gt[:int(ood_len * ratio)]])
        elif self.config.discoverer.ood_type == 'ViM':
            from numpy.linalg import norm, pinv
            from scipy.special import logsumexp
            from sklearn.covariance import EmpiricalCovariance
            DIM = id_feat.shape[1] // 2
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(id_feat.cpu())
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            NS = np.ascontiguousarray(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
            vlogit_id_train = norm(np.matmul(id_feat.cpu(), NS), axis=-1)
            alpha = id_logit.max(axis=-1)[0].mean() / vlogit_id_train.mean()
            print(f'Computing *** ViM *** metrics on OOD (new-train) dataset...')
            # print(f'DIM={DIM}, alpha={alpha:.4f}')

            id_energy = logsumexp(id_logit.cpu(), axis=-1)
            ood_energy = logsumexp(ood_logit.cpu(), axis=-1)
            id_vlogit = norm(np.matmul(id_feat.numpy(), NS), axis=-1) * alpha.cpu().numpy()
            ood_vlogit = norm(np.matmul(ood_feat.numpy(), NS), axis=-1) * alpha.cpu().numpy()

            tid_conf = torch.tensor(-id_vlogit + id_energy)
            tood_conf = torch.tensor(-ood_vlogit + ood_energy)

            conf = torch.cat([tid_conf[:int(id_len * ratio)], tood_conf[:int(ood_len * ratio)]])
            label = np.concatenate([id_gt[:int(id_len * ratio)], ood_neg_gt[:int(ood_len * ratio)]])
        elif self.config.discoverer.ood_type == 'DML':
            w = net.fc.weight.clone().detach()
            w = F.normalize(w, p=2, dim=1).cpu()
            w = w[::2, ]  # savc使用的是两倍类别数的fc层
            id_cosine = F.normalize(id_feat, p=2, dim=1) @ w.T
            ood_cosine = F.normalize(ood_feat, p=2, dim=1) @ w.T
            id_mcos, _ = torch.max(id_cosine, dim=1, keepdim=True)
            ood_mcos, _ = torch.max(ood_cosine, dim=1, keepdim=True)
            id_norm = torch.norm(id_feat, dim=1)
            ood_norm = torch.norm(ood_feat, dim=1)
            tid_conf = id_mcos + 0.002 * id_norm.unsqueeze(1)
            tood_conf = ood_mcos + 0.002 * ood_norm.unsqueeze(1)
            conf = torch.cat([tid_conf[:int(id_len * ratio)], tood_conf[:int(ood_len * ratio)]])
            label = np.concatenate([id_gt[:int(id_len * ratio)], ood_neg_gt[:int(ood_len * ratio)]])

        elif self.config.discoverer.ood_type == 'DMLplus':
            w = net.fc.weight.clone().detach()
            w = F.normalize(w, p=2, dim=1).cpu()
            w = w[::2, ]  # savc使用的是两倍类别数的fc层
            id_cosine = F.normalize(id_feat, p=2, dim=1) @ w.T
            ood_cosine = F.normalize(ood_feat, p=2, dim=1) @ w.T
            id_mcos, _ = torch.max(id_cosine, dim=1, keepdim=True)
            ood_mcos, _ = torch.max(ood_cosine, dim=1, keepdim=True)
            id_norm = torch.norm(id_feat, dim=1)
            ood_norm = torch.norm(ood_feat, dim=1)
            tid_conf = id_mcos + 0.002 * id_norm.unsqueeze(1)
            tood_conf = ood_mcos + 0.002 * ood_norm.unsqueeze(1)
            ttood_conf = F.normalize(ood_logit, p=2, dim=1) @ prototype_cls.T
            ttood_conf, _ = torch.max(ttood_conf, dim=1, keepdim=True)
            ttid_conf = id_conf
            tid_conf = tid_conf + 40 * ttid_conf.cpu()
            tood_conf = tood_conf + 40 * ttood_conf.cpu()
            conf = torch.cat([tid_conf[:int(id_len * ratio)], tood_conf[:int(ood_len * ratio)]])
            label = np.concatenate([id_gt[:int(id_len * ratio)], ood_neg_gt[:int(ood_len * ratio)]])
        elif self.config.discoverer.ood_type == 'PatternMatching':
            conf = torch.cat([id_conf[:int(id_len*ratio)], ood_conf[:int(ood_len*ratio)]])
            label = np.concatenate([id_gt[:int(id_len*ratio)], ood_neg_gt[:int(ood_len*ratio)]])
        elif self.config.discoverer.ood_type == 'classification':
            conf = torch.cat([id_conf[:int(id_len*ratio)], ood_conf[:int(ood_len*ratio)]])
            label = np.concatenate([id_gt[:int(id_len*ratio)], ood_neg_gt[:int(ood_len*ratio)]])


        conf = conf.view(-1)
        # threshold, ood_metric = self._search_threshold(conf, label)
        # print(f"Finish searching threshold on val-set: {threshold.item()}, OOD samples Percision: {ood_metric['precision']}, Recall: {ood_metric['recall']}",flush=True)

        # --------- Using threshold to get the novel samples ----------- #
        total_feat = torch.cat((id_feat, ood_feat), dim=0)
        if self.config.discoverer.ood_type == 'MLS':
            tid_conf, _ = torch.max(id_logit.cpu(), dim=1, keepdim=True)
            tood_conf, _ = torch.max(ood_logit.cpu(), dim=1, keepdim=True)
            conf = torch.cat([tid_conf, tood_conf])
            label = np.concatenate([id_gt, ood_gt])
        elif self.config.discoverer.ood_type == 'MSP':
            tmp = torch.softmax(id_logit.cpu(), 1)
            tid_conf, _ = torch.max(tmp, dim=1, keepdim=True)
            tmp = torch.softmax(ood_logit.cpu(), 1)
            tood_conf, _ = torch.max(tmp, dim=1, keepdim=True)
            conf = torch.cat([tid_conf, tood_conf])
            label = np.concatenate([id_gt, ood_gt])
        elif self.config.discoverer.ood_type == 'MCM':
            T = 2
            tmp = torch.softmax(id_logit.cpu() / T, 1)
            tid_conf, _ = torch.max(tmp, dim=1, keepdim=True)
            tmp = torch.softmax(ood_logit.cpu() / T, 1)
            tood_conf, _ = torch.max(tmp, dim=1, keepdim=True)
            conf = torch.cat([tid_conf, tood_conf])
            label = np.concatenate([id_gt, ood_gt])
        elif self.config.discoverer.ood_type == 'Energy':
            tid_conf = torch.tensor(logsumexp(id_logit.cpu(), axis=-1))
            tood_conf = torch.tensor(logsumexp(ood_logit.cpu(), axis=-1))
            conf = torch.cat([tid_conf, tood_conf])
            label = np.concatenate([id_gt, ood_gt])
        elif self.config.discoverer.ood_type == 'ViM':
            from numpy.linalg import norm, pinv
            from scipy.special import logsumexp
            from sklearn.covariance import EmpiricalCovariance
            DIM = id_feat.shape[1] // 2
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(id_feat.cpu())
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            NS = np.ascontiguousarray(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
            vlogit_id_train = norm(np.matmul(id_feat.cpu(), NS), axis=-1)
            alpha = id_logit.max(axis=-1)[0].mean() / vlogit_id_train.mean()
            print(f'Computing *** ViM *** metrics on OOD (new-train) dataset...')
            # print(f'DIM={DIM}, alpha={alpha:.4f}')

            id_energy = logsumexp(id_logit.cpu(), axis=-1)
            ood_energy = logsumexp(ood_logit.cpu(), axis=-1)
            id_vlogit = norm(np.matmul(id_feat.numpy(), NS), axis=-1) * alpha.cpu().numpy()
            ood_vlogit = norm(np.matmul(ood_feat.numpy(), NS), axis=-1) * alpha.cpu().numpy()

            tid_conf = torch.tensor(-id_vlogit + id_energy)
            tood_conf = torch.tensor(-ood_vlogit + ood_energy)
            conf = torch.cat([tid_conf, tood_conf])
            label = np.concatenate([id_gt, ood_gt])
        elif self.config.discoverer.ood_type == 'DML':
            w = net.fc.weight.clone().detach()
            w = F.normalize(w, p=2, dim=1).cpu()
            w = w[::2, ]  # savc使用的是两倍类别数的fc层
            id_cosine = F.normalize(id_feat, p=2, dim=1) @ w.T
            ood_cosine = F.normalize(ood_feat, p=2, dim=1) @ w.T
            id_mcos, _ = torch.max(id_cosine, dim=1, keepdim=True)
            ood_mcos, _ = torch.max(ood_cosine, dim=1, keepdim=True)
            id_norm = torch.norm(id_feat, dim=1)
            ood_norm = torch.norm(ood_feat, dim=1)

            tid_conf = id_mcos + 0.002 * id_norm.unsqueeze(1)
            tood_conf = ood_mcos + 0.002 * ood_norm.unsqueeze(1)

            conf = torch.cat([tid_conf, tood_conf])
            label = np.concatenate([id_gt, ood_gt])
        elif self.config.discoverer.ood_type == 'DMLplus':

            w = net.fc.weight.clone().detach()
            w = F.normalize(w, p=2, dim=1).cpu()
            w = w[::2, ]  # savc使用的是两倍类别数的fc层
            id_cosine = F.normalize(id_feat, p=2, dim=1) @ w.T
            ood_cosine = F.normalize(ood_feat, p=2, dim=1) @ w.T
            id_mcos, _ = torch.max(id_cosine, dim=1, keepdim=True)
            ood_mcos, _ = torch.max(ood_cosine, dim=1, keepdim=True)
            id_norm = torch.norm(id_feat, dim=1)
            ood_norm = torch.norm(ood_feat, dim=1)
            tid_conf = id_mcos + 0.002 * id_norm.unsqueeze(1)
            tood_conf = ood_mcos + 0.002 * ood_norm.unsqueeze(1)
            ttood_conf = F.normalize(ood_logit, p=2, dim=1) @ prototype_cls.T
            ttood_conf, _ = torch.max(ttood_conf, dim=1, keepdim=True)
            ttid_conf = id_conf
            tid_conf = tid_conf + 40 * ttid_conf.cpu()
            tood_conf = tood_conf + 40 * ttood_conf.cpu()
            conf = torch.cat([tid_conf, tood_conf])
            label = np.concatenate([id_gt, ood_gt])
        elif self.config.discoverer.ood_type == 'PatternMatching':
            conf = torch.cat([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])  # Using the number label not -1
        elif self.config.discoverer.ood_type == 'classification':
            conf = torch.cat([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])  # Using the number label not -1

        total_imgpath_list = id_imgpath_list + ood_imgpath_list
        conf = conf.view(-1)

        novel_imgpath_list, novel_target, pseudo_labels = self._get_pseudo(conf, label, ood_gt, threshold, total_imgpath_list, total_feat)
        count = sum(p1 == p2 for p1, p2 in zip(pseudo_labels, novel_target))
        ncd_acc = count / len(pseudo_labels)
        print(f"Finish NCD! OOD samples ACC: {ncd_acc}", flush=True)

        # --------- Update the ood dataloader as the ncd trainloader ----------- #
        ood_data_loader.dataset.data, ood_data_loader.dataset.targets = novel_imgpath_list, pseudo_labels
        ood_data_loader.dataset.transform = train_transfrom

        return ood_data_loader

    # def _search_threshold(self, conf, label):
    #     print(f'Searching the threshold for ood ...', flush=True)
    #     conf = np.array(conf)
    #     label = np.array(label)
    #     ood_indicator = np.zeros_like(label)
    #     ood_indicator[label == -1] = 1
    #     precisions, recalls, thresholds = metrics.precision_recall_curve(ood_indicator, -conf)

    #     idx = np.argmax(precisions > self.config.discoverer.precision_bar)
    #     best_threshold = thresholds[idx]

    #     plt.figure()
    #     plt.plot(thresholds, precisions[:-1])
    #     plt.title('Precision-Threshold')
    #     plt.xlabel('Threshold')
    #     plt.ylabel('Precision')
    #     plt.savefig(os.path.join(self.config.output_dir, 'Pr_Trd.png'))

    #     plt.figure()
    #     plt.plot(recalls, precisions)
    #     plt.title('P-R')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.savefig(os.path.join(self.config.output_dir, 'P-R.png'))

    #     ood_metric = {'precision': precisions[idx], 'recall': recalls[idx]}

    #     return torch.tensor(-best_threshold), ood_metric

    def _get_pseudo(self, conf, label, ood_gt, threshold, total_imgpath_list, total_feat):

        # 使用 t-SNE 对特征进行降维
        tsne = TSNE(n_components=2, random_state=0)
        features_tsne = tsne.fit_transform(total_feat)

        # 绘制结果
        plt.figure(figsize=(10, 6))
        unique_labels = np.unique(label)
        cmap = plt.get_cmap('tab20')
        for i, l in enumerate(unique_labels):
            indices = label == l
            plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], label=l, c=[cmap(i)])

        plt.legend()
        plt.title('t-SNE visualization of features')
        plt.xlabel('t-SNE axis 1')
        plt.ylabel('t-SNE axis 2')
        plt.savefig(os.path.join(self.config.output_dir, 'tSNE.png'))


        novel_indices = torch.nonzero(conf < threshold)[:, 0].squeeze()
        novel_indices_list = novel_indices.tolist()
        assert isinstance(novel_indices_list, list) or len(novel_indices_list) == 1, "novel_indices wrong"

        novel_imgpath_list = [total_imgpath_list[i] for i in novel_indices_list]
        novel_target = label[novel_indices]  # 真实的新标签
        novel_feat = total_feat[novel_indices]
        kmeans = KMeans(self.config.dataloader.way)
        kmeans.fit(novel_feat)
        pseudo_labels = kmeans.labels_  # 每个样本的聚类标签

        max_similarities = []
        max_neighbors = []
        # print(pseudo_labels)
        for i in range(novel_feat.size(0)): # 咩有考虑到只有一个样本的情况
            max_similarity = 0.0
            index = 0
            for j in range(novel_feat.size(0)):
                if i != j and pseudo_labels[i]==pseudo_labels[j]:
                    similarity = cosine_similarity(novel_feat[i].unsqueeze(0), novel_feat[j].unsqueeze(0))
                    if similarity > max_similarity:
                        index = j
                        max_similarity = similarity
            if max_similarity == 0.0:
                max_similarities.append(1.0)
            else:
                max_similarities.append(max_similarity[0][0])
            max_neighbors.append(index)
        sift_threshold = self.config.discoverer.sift_threshold
        max_similarities = np.array(max_similarities)
        if np.min(novel_target) >= np.min(ood_gt): # all samples are new class
            sift_threshold = np.min(max_similarities) - 1e-8
        else:
            sift_threshold = np.max(max_similarities[novel_target < np.min(ood_gt)])+1e-8

        sift_indices = torch.tensor([i for i in range(len(max_similarities)) \
                                     if max_similarities[i] > sift_threshold])
        if sift_indices.numel() >0:
            sift_indices = torch.tensor([i for i in range(len(max_similarities))])
        # assert sift_indices.numel() > 0, "There is no novel samples."
        novel_target = novel_target[sift_indices] # 真实的新标签
        novel_feat = novel_feat[sift_indices]

        sift_indices = sift_indices.tolist()
        novel_imgpath_list = [novel_imgpath_list[i] for i in sift_indices]

        kmeans = KMeans(self.config.dataloader.way)
        kmeans.fit(novel_feat)
        pseudo_labels = kmeans.labels_  # 每个样本的聚类标签
        pseudo_labels = self._split_cluster_label(novel_target, pseudo_labels, ood_gt)
        print(pseudo_labels, novel_target, flush=True)
        return novel_imgpath_list, novel_target, pseudo_labels

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
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
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

    def report(self, test_metrics):
        print('Completed!', flush=True)

    def _split_cluster_label(self, y_label, y_pred, ood_class):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        First compute linear assignment on all data, then look at how good the accuracy is on subsets

        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`

        # Return
            cluster_label: cluster label
        """
        y_label = y_label.astype(int)
        assert y_pred.size == y_label.size

        index = np.array([i for i in range(len(y_label)) if y_label[i] in ood_class])
        y_label_crop = y_label[index]
        y_pred_crop = y_pred[index]
        
        D = max(y_pred_crop.max(), y_label_crop.max()) + 1
        w = np.zeros((D, D), dtype=int)
        for i in range(y_pred_crop.size):
            w[y_pred_crop[i], y_label_crop[i]] += 1

        ind = linear_assignment(w.max() - w)
        ind = np.vstack(ind).T

        cluster_label = np.array([ind[x, 1] for x in y_pred])

        if np.setdiff1d(cluster_label, ood_class).size > 0:
            cluster_label_diff = np.setdiff1d(cluster_label, ood_class)
            ood_class_diff = np.setdiff1d(ood_class, cluster_label)
            for i in range(cluster_label_diff.size):
                cluster_label[cluster_label == cluster_label_diff[i]] = ood_class_diff[i]

        return cluster_label