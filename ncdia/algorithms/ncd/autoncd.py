from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

from ncdia.algorithms.ood import AutoOOD


class AutoNCD(object):
    """AutoNCD class for evaluating with OOD metrics and 
    relabeling the OOD dataset for the next session.

    Args:
        model (nn.Module): model to be evaluated
        train_loader (DataLoader): train dataloader
        test_loader (DataLoader): test dataloader
        device (torch.device, optional): device to run the evaluation. Default to None.
        verbose (bool, optional): print the progress bar. Default to False.
    
    """
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            test_loader: DataLoader,
            device: torch.device = None,
            verbose: bool = False,
    ):
        super(AutoNCD, self).__init__()
        self.model = model.eval()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.verbose = verbose

        if not device:
            for param in self.model.parameters():
                device = param.device
                break
        self.device = device
        
    @torch.no_grad()
    def inference(
            self,
            dataloader: DataLoader,
            split: str = 'train',
    ):
        """Inference the model on the dataloader and return relevant information.
        If split is 'train', return the prototype of the training data.

        Args:
            dataloader (DataLoader): dataloader for evaluation
            split (str, optional): train or test. Defaults to 'train'.

        Returns:
            If split is 'train':
                features (np.ndarray): feature vectors, (N, D)
                logits (np.ndarray): logit vectors, (N, C)
                prototype_cls (np.ndarray): prototype vectors, (C, D)
            If split is 'test':
                imgpaths (list): image paths (list)
                features (np.ndarray): feature vectors, (N, D)
                logits (np.ndarray): logit vectors, (N, C)
                preds (np.ndarray): prediction labels, (N,)
                labels (np.ndarray): ground truth labels, (N,)
        """
        imgpaths, features, logits, preds, confs, labels = [], [], [], [], [], []
        
        tbar = tqdm(dataloader, dynamic_ncols=True, disable=not self.verbose)
        for batch in tbar:
            data = batch['data'].to(self.device)
            label = batch['label'].to(self.device)
            attribute = batch['attribute'].to(self.device)
            imgpath = batch['imgpath']

            joint_preds = self.model(data)
            joint_preds = joint_preds[:, :self.all_classes]
            score = torch.softmax(joint_preds, dim=1)
            conf, pred = torch.max(score, dim=1)
            feats = self.model.get_features(data)

            imgpaths += imgpath
            features.append(feats.detach().clone().cpu())
            logits.append(joint_preds.detach().clone().cpu())
            preds.append(pred.detach().clone().cpu())
            confs.append(conf.detach().clone().cpu())
            labels.append(label.cpu())
        
        # convert values into numpy array
        features = torch.cat(features, dim=0).numpy()
        logits = torch.cat(logits, dim=0).numpy()
        preds = torch.cat(preds, dim=0).numpy().astype(int)
        confs = torch.cat(confs, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy().astype(int)

        if split == 'train':
            labels = torch.tensor(labels)
            classes = torch.unique(labels)
            prototype_cls = []

            for cls in classes:
                cls_indices = torch.where(labels == cls)
                cls_preds = logits[cls_indices]
                prototype_cls.append(torch.mean(cls_preds, dim=0))
            
            return features, logits, torch.stack(prototype_cls)

        return imgpaths, features, logits, preds, labels
    
    def relabel(
            self,
            ood_loader: DataLoader,
            metrics: list = [],
            tpr_th: float = 0.95,
            prec_th: float = None,
    ):
        """Relabel the OOD dataset for the next session.

        Args:
            ood_loader (DataLoader): OOD dataloader for relabeling
            metrics (list, optional): metrics to evaluate the OOD dataset. Defaults to [].
            tpr_th (float, optional): True positive rate threshold. Defaults to 0.95.
            prec_th (float, optional): Precision threshold. Defaults to None.

        Returns:
            DataLoader: relabeled OOD dataloader
        """
        assert prec_th is not None, "Precision threshold must be provided."
        if not isinstance(metrics, list):
            metrics = [metrics]

        # weight of the fully connected layer of the model
        self.fc_weight = self.model.fc.weight.detach().clone().cpu()

        self.ood_loader = ood_loader
        self.base_classes = self.train_loader.dataset.num_classes
        self.inc_classes = self.ood_loader.dataset.num_classes
        self.all_classes = self.base_classes + self.inc_classes

        # prepare prototype of training data
        self.train_feats, self.train_logits, self.prototype_cls = \
            self.inference(self.train_loader, split='train')
        self.prototype_cls = F.normalize(self.prototype_cls, p=2, dim=1)

        # prepare id statistics from test data
        self.id_imgpaths, self.id_feats, self.id_logits, self.id_preds, self.id_labels = \
            self.inference(self.test_loader, split='test')

        # prepare ood statistics from ood data
        self.ood_imgpaths, self.ood_feats, self.ood_logits, self.ood_preds, self.ood_labels = \
            self.inference(self.ood_loader, split='test')
        
        self.ood_metrics = AutoOOD._eval(
            self.prototype_cls, self.fc_weight,
            self.train_feats, self.train_logits,
            self.id_feats, self.id_logits, self.id_labels,
            self.ood_feats, self.ood_logits, self.ood_labels,
            metrics=metrics, tpr_th=tpr_th, prec_th=prec_th,
        )

        metric = metrics[0]
        threshold = self.ood_metrics[metric][1][0]
        conf = self.ood_metrics[metric][1][1]
        label = self.ood_metrics[metric][1][2]

        total_imgpath_list = self.id_imgpaths + self.ood_imgpaths
        total_feat = torch.cat((self.id_feats, self.ood_feats), dim=0)

        novel_indices = torch.nonzero(conf < threshold)[:, 0].squeeze()
        novel_indices_list = novel_indices.tolist()
        novel_imgpath_list = [total_imgpath_list[i] for i in novel_indices_list]
        novel_target = label[novel_indices]
        novel_feat = total_feat[novel_indices]

        kmeans = KMeans(self.inc_classes)
        kmeans.fit(novel_feat)
        pseudo_labels = kmeans.labels_

        max_neighbors, max_similarities = [], []
        for i in range(novel_feat.size(0)):
            index, max_similarity = 0, 0.0
            for j in range(novel_feat.size(0)):
                if i != j and pseudo_labels[i]==pseudo_labels[j]:
                    similarity = cosine_similarity(novel_feat[i].unsqueeze(0), novel_feat[j].unsqueeze(0))
                    if similarity > max_similarity:
                        index, max_similarity = j, similarity
            if max_similarity == 0.0:
                max_similarities.append(1.0)
            else:
                max_similarities.append(max_similarity[0][0])
            max_neighbors.append(index)
        max_similarities = np.array(max_similarities)

        if np.min(novel_target) >= np.min(self.ood_labels): # all samples are new class
            sift_threshold = np.min(max_similarities) - 1e-8
        else:
            sift_threshold = np.max(max_similarities[novel_target < np.min(self.ood_labels)]) + 1e-8
        sift_indices = torch.tensor([i for i in range(len(max_similarities)) \
                                     if max_similarities[i] > sift_threshold])
        
        if sift_indices.numel()==0:
            sift_indices = torch.tensor([i for i in range(len(max_similarities))])
        assert sift_indices.numel() > 0, "There is no novel samples."
        novel_target = novel_target[sift_indices]
        novel_feat = novel_feat[sift_indices]

        sift_indices = sift_indices.tolist()
        novel_imgpath_list = [novel_imgpath_list[i] for i in sift_indices]

        kmeans = KMeans(self.inc_classes)
        kmeans.fit(novel_feat)
        pseudo_labels = kmeans.labels_
        pseudo_labels = self._split_cluster_label(novel_target, pseudo_labels, self.ood_labels)

        ood_loader.dataset.data = novel_imgpath_list
        ood_loader.dataset.targets = pseudo_labels

        return ood_loader

    def _split_cluster_label(self, y_label, y_pred, ood_class):
        """Calculate clustering accuracy. Require scikit-learn installed
        First compute linear assignment on all data, then look at how good the accuracy is on subsets

        Args:
            y_label: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
            ood_class: out-of-distribution class labels

        Returns:
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

        ind = linear_sum_assignment(w.max() - w)
        ind = np.vstack(ind).T

        cluster_label = np.array([ind[x, 1] for x in y_pred])

        if np.setdiff1d(cluster_label, ood_class).size > 0:
            cluster_label_diff = np.setdiff1d(cluster_label, ood_class)
            ood_class_diff = np.setdiff1d(ood_class, cluster_label)
            for i in range(cluster_label_diff.size):
                cluster_label[cluster_label == cluster_label_diff[i]] = ood_class_diff[i]

        return cluster_label
