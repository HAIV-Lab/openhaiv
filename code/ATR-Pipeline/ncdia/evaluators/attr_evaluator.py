import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize

import ncdia.utils.comm as comm
from .metrics import BasePostprocessor
from ncdia.utils import Config


def to_np(x):
    return x.data.cpu().numpy()


class AttrEvaluator:
    def __init__(self, config: Config):
        self.config = config

    def eval_acc_attr(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1,
                 threshold: float = 0.5):
        net.eval()

        # 初始化列表来存储每个样本的真实和预测标签
        true_attrs = []
        pred_attrs = []
        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True,
                              disable=not comm.is_main_process()):
                # prepare data
                data = batch['data'].cuda()
                target = batch['label'].cuda()
                attributes = batch['attribute'].cuda()

                # forward
                output, attr_output = net(data)

                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

                # 应用sigmoid函数并使用阈值
                pred_attrs_batch = (torch.sigmoid(attr_output) > threshold).type(torch.int).cpu().numpy()
                true_attrs.append(attributes.cpu().numpy())
                pred_attrs.append(pred_attrs_batch)


        loss = loss_avg / len(data_loader)
        acc = correct / len(data_loader.dataset)

        # 累积指标计算
        true_attrs = np.concatenate(true_attrs)
        pred_attrs = np.concatenate(pred_attrs)

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
            'loss': self.save_metrics(loss),
            'OP': self.save_metrics(OP),
            'OR': self.save_metrics(OR),
            'OF1': self.save_metrics(OF1),
            'CP': self.save_metrics(CP),
            'CR': self.save_metrics(CR),
            'CF1': self.save_metrics(CF1),
            'mAP': self.save_metrics(mAP),
        }
        return metrics

    def extract(self,
                net: nn.Module,
                data_loader: DataLoader,
                filename: str = 'feature'):
        net.eval()
        feat_list, label_list = [], []

        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Feature Extracting: ',
                              position=0,
                              leave=True,
                              disable=not comm.is_main_process()):
                data = batch['data'].cuda()
                label = batch['label']

                _, feat = net(data, return_feature=True)
                feat_list.extend(to_np(feat))
                label_list.extend(to_np(label))

        feat_list = np.array(feat_list)
        label_list = np.array(label_list)

        save_dir = self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, filename),
                 feat_list=feat_list,
                 label_list=label_list)

    def save_metrics(self, value):
        all_values = comm.gather(value)
        temp = 0
        for i in all_values:
            temp = temp + i
        # total_value = np.add([x for x in all_values])s

        return temp
