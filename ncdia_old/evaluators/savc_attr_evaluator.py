import csv
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .metrics import BasePostprocessor
from ncdia_old.utils import Config
import ncdia_old.utils.comm as comm

from .base_evaluator import BaseEvaluator
from .metrics import compute_ood_metrics

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

from ncdia_old.augmentations import fantasy
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
                  id_split: str = 'train',
                  is_draw_confusion = False):
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
        
        # draw confusion matrix
        # if is_draw_confusion:
        #     if session == 0:
        #         draw_confusion_matrix(label_true=label_list, label_pred=pred_list,
        #                             label_name=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
        #                             title="混淆矩阵",
        #                             pdf_save_path=os.path.join(self.config.output_dir, "Confusion_Matrix_Base.jpg"),
        #                             dpi=300)
        #     else:
        #         draw_confusion_matrix(label_true=label_list, label_pred=pred_list,
        #                             label_name=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
        #                             title="混淆矩阵",
        #                             pdf_save_path=os.path.join(self.config.output_dir, "Confusion_Matrix_Inc.jpg"),
        #                             dpi=300)
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

            return prototype_cls, prototype_att

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
        prototype_cls, prototype_att = self.inference(
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
        self._eval_ood(net, [id_feat, id_conf, id_gt, id_att_conf, id_att_gt, prototype_cls, prototype_att],
                       ood_data_loaders, session)


    def _eval_ood(self,
                  net: nn.Module,
                  id_list: List[np.ndarray],
                  ood_data_loader,
                  session: int = -1):
        print(f'Processing ood inference...', flush=True)
        [id_feat, id_conf, id_gt, id_att_conf, id_att_gt, prototype_cls, prototype_att] = id_list
        metrics_list = []

        ood_feat, ood_pred, ood_logit, ood_gt, ood_att_pred, ood_att_logit, ood_att_gt = \
            self.inference(net, ood_data_loader, session=session, id_split='test')
        ood_logit = ood_logit.clone().detach()
        ood_att_logit = ood_att_logit.clone().detach()

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

        ood_conf = F.normalize(ood_logit, p=2, dim=1) @ prototype_cls.T
        ood_att_conf = F.normalize(ood_att_logit, p=2, dim=1) @ prototype_att.T
        ood_conf, _ = torch.max(ood_conf, dim=1, keepdim=True)
        ood_att_conf, _ = torch.max(ood_att_conf, dim=1, keepdim=True)
        ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood

        if self.config.recorder.save_scores:
            self._save_scores(ood_pred, ood_conf, ood_gt, 'OOD')

        # pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([id_conf.cpu(), ood_conf.cpu()])
        label = np.concatenate([id_gt, ood_gt])

        print(f'Computing metrics on OOD (new-train) dataset...')
        ood_metrics_cls = compute_ood_metrics(conf, label) #, pred)

        conf = np.concatenate([id_att_conf.cpu(), ood_att_conf.cpu()])
        label = np.concatenate([id_gt, ood_gt])
        ood_metrics_att = compute_ood_metrics(conf, label)

        conf = np.concatenate([id_att_conf.cpu() + id_conf.cpu(), \
                               ood_att_conf.cpu() + ood_conf.cpu()])
        label = np.concatenate([id_gt, ood_gt])
        ood_metrics_merge = compute_ood_metrics(conf, label)
    
        if self.config.recorder.save_csv:
            self._save_csv(ood_metrics_cls, dataset_name='OOD_cls')
            self._save_csv(ood_metrics_att, dataset_name='OOD_att')
            self._save_csv(ood_metrics_merge, dataset_name='OOD_merge')

        metrics_list.append(ood_metrics_cls)

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name='OOD_cls')


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
            net, data_loader, session=session, id_split='test', is_draw_confusion=True)
        
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


# 
def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path="./confusion_matrix", dpi=300):
    """
    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:
    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()
    
    font_size = 8
    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            if value >0.02:
                plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color, fontsize=font_size)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
        plt.clf()

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

            save_directory = '/new_data/dx450/ATR-13-pingzhuang/'+str(cls)
            os.makedirs(save_directory, exist_ok=True)
            import shutil
            for img_path in merge_imgpaths:
                save_path = os.path.join(save_directory, os.path.basename(img_path))
                shutil.copy2(img_path, save_path)

