import torch.nn as nn
from tqdm import tqdm

from ncdia.utils import TRAINERS, Configs
from ncdia.dataloader import MergedDataset
from .pretrainer import PreTrainer
from .hooks import NCDHook
from ultralytics import YOLO
import os
import torch
from PIL import Image
import torch.nn.functional as F

import xml.etree.ElementTree as ET
# from ncdia.algorithms.ncd import AutoNCD
from ncdia.algorithms.ood import AutoOOD
from torch.utils.data import DataLoader

@TRAINERS.register
class Inference(PreTrainer):
    """IncTrainer class for incremental training.

    Args:
        model (nn.Module): Model to be trained.
        cfg (dict, optional): Configuration for trainer.
        sess_cfg (Configs): Session configuration.
        session (int, optional): Session number. Default: 0.

    Attributes:
        sess_cfg (Configs): Session configuration.
        num_sess (int): Number of sessions.
        session (int): Session number. If == 0, execute pre-training.
            If > 0, execute incremental training.
        hist_trainset (MergedDataset): Historical training dataset.
        hist_valset (MergedDataset): Historical validation dataset.
        hist_testset (MergedDataset): Historical testing dataset.

    """

    def __init__(
            self,
            cfg: dict | None = None,
            sess_cfg: Configs | None = None,
            ncd_cfg: Configs | None = None,
            session: int = 0,
            model: nn.Module = None,
            hist_trainset: MergedDataset = None,
            hist_testset: MergedDataset = None,
            **kwargs
    ) -> None:
        self.sess_cfg = sess_cfg
        self.num_sess = len(sess_cfg.keys())
        self.ncd_cfg = ncd_cfg
        self.kwargs = kwargs

        s_cfg = sess_cfg[f's{session}'].cfg
        cfg.merge_from_config(s_cfg)
        cfg.freeze()

        # Specify historical datasets to store previous data
        if not hist_trainset:
            hist_trainset = MergedDataset()
        self.hist_trainset = hist_trainset

        if not hist_testset:
            hist_testset = MergedDataset()
        self.hist_testset = hist_testset



        super(Inference, self).__init__(
            cfg=cfg,
            session=session,
            model=model,
            max_epochs=0,
            custom_hooks=[],
            **self.kwargs
        )
        img_list, gt_list, pred_obj_num = self.generate_classify_datasets()
        self.test_loader.dataset.data = img_list
        self.test_loader.dataset.targets = gt_list
        # print(len(self.test_loader.dataset.data))


    def train(self) -> nn.Module:
        """Incremental training.
        `self.num_sess` determines the number of sessions,
        and session number is stored in `self.session`.

        Returns:
            model (nn.Module): Trained model.
        """
        self.load_ckpt(self.cfg['ckpt'][self.session])
        for session in range(self.num_sess):

            if session > 0:
                new_instance = Inference(
                    cfg=self.cfg,
                    sess_cfg=self.sess_cfg,
                    ncd_cfg=self.ncd_cfg,
                    session=session,
                    model=self.model,
                    hist_trainset=self.hist_trainset,
                    hist_testset=self.hist_testset,
                    **self.kwargs
                )
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__


            super(Inference, self).train()

            # Store historical data
            if session==0:
                self.hist_testset.labels = self.test_loader.dataset.targets
                self.hist_testset.images = self.test_loader.dataset.data
                self.hist_testset.transform = self.test_loader.dataset.transform
            else:
                self.hist_testset.labels += self.test_loader.dataset.targets
                self.hist_testset.images += self.test_loader.dataset.data
                self.hist_testset.transform = self.test_loader.dataset.transform
            if session>0:
                self.eval_ood()

        return self.model

    def eval_ood(self):
        metrics = self.ncd_cfg.metrics or ['msp']
        threshold = self.cfg['threshold']

        fc_weight = self.model.fc.weight.clone().detach().cpu()
        filename = 'train_static.pt'
        static_path = os.path.join(self.work_dir, filename)
        train_static = torch.load(static_path)
        train_logits = train_static['train_logits']
        logits_att = train_static['att_logits']
        train_feat = train_static['train_features']
        prototype = train_static['prototype']
        prototype_att = train_static['prototype_att']
        prototype = F.normalize(prototype, p=2, dim=1)
        prototype_att = F.normalize(prototype_att, p=2, dim=1)
        logits, feat, label = self.inference()

        conf = AutoOOD.inference(metrics, logits, feat, train_logits, train_feat, fc_weight, prototype, logits_att, prototype_att)



        novel_indices = torch.nonzero(conf[metrics[0]] < threshold)[:, 0]
        new_class = list(set(self.train_loader.dataset.labels))
        acc_num_k,acc_num_uk,num_k,num_uk = 0,0,0,0
        for i in range(len(self.hist_testset.labels)):
            if self.hist_testset.labels[i] in new_class and i in novel_indices:
                acc_num_uk += 1
            if self.hist_testset.labels[i] not in new_class and i not in novel_indices:
                # if self.hist_testset.labels[i] == :
                    acc_num_k += 1
            if self.hist_testset.labels[i] in new_class :
                num_uk += 1
            if  self.hist_testset.labels[i] not in new_class :
                num_k += 1
        # ood_acc = acc_num / len(self.hist_testset.labels)
        print('ood_acc_k', acc_num_k/num_k)

        print('ood_acc_uk', acc_num_uk/num_uk)

    def inference(self):
        all_class = self.hist_testset.num_classes
        default_loader = {
            'batch_size': 64,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
        }
        loader_cfg = default_loader
        ncd_cfg = self.ncd_cfg
        loader_cfg.update(ncd_cfg.dataloader or {})
        dataloader = DataLoader(self.hist_testset, **loader_cfg)

        features, logits, labels = [], [], []

        tbar = tqdm(dataloader, dynamic_ncols=True, disable=True)
        for batch in tbar:
            data = batch['data'].to(self.device)
            label = batch['label'].to(self.device)
            joint_preds = self.model(data)
            if isinstance(joint_preds, tuple):
                joint_preds, _ = joint_preds
            joint_preds = joint_preds[:, :all_class]
            feats = self.model.get_features(data)

            features.append(feats.clone().detach().cpu())
            logits.append(joint_preds.clone().detach().cpu())
            labels.append(label.clone().detach().cpu())

        features = torch.cat(features, dim=0)
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0).to(torch.int)

        return logits, features, labels


    def generate_classify_datasets(self):

        include_cls = ['T-62', 'T-62M', 'T-62M-1', 'T-72A', 'T-72AB', 'T-72AV__TURMS-T_', 'T-72B', 'T-72B3', 'T-72B3_',
                       'T-72B_1989_', 'T-72M2_xdh', '_R_T-62A', '_T-62_545', 'T-62A', 'T-62A_', '0_T-72M1', '1_T-72M1',
                       '2_T-72M1']

        # include_cls += ['Nimitz', 'Arleigh Burke', 'Arleigh Burke DD', 'Asagiri', 'Asagiri DD', 'Midway']
        #
        #
        #
        # include_cls += ['A8', 'B-52', 'A10', 'B-1B', 'A15', 'F-22']

        # include_cls += ['BM-8-24','BM-13N', 'BMP-2', 'BMP-2M', 'T34_rocket', 'B-133-new']

        map_dict = {'Nimitz': 0, 'BM-8-24': 1, 'BM-13N': 2, 'Midway': 3, 'Arleigh Burke': 4, 'Arleigh Burke DD': 4,
                    'Asagiri': 5,
                    'Asagiri DD': 5, 'A8': 6, 'B-52': 6, 'A10': 7, 'B-1B': 7, 'BMP-2': 8, 'BMP-2M': 8,
                    'T-62': 9, 'T-62M': 9, 'T-62M-1': 9, 'T-72A': 9, 'T-72AB': 9, 'T-72AV__TURMS-T_': 9, 'T-72B': 9,
                    'T-72B3': 9, 'T-72B3_': 9, 'T-72B_1989_': 9, 'T-72M2_xdh': 9, '_R_T-62A': 9, '_T-62_545': 9,
                    'T-62A': 9, 'T-62A_': 9, '0_T-72M1': 9, '1_T-72M1': 9, '2_T-72M1': 9,
                    'T34_rocket': 10,
                    'A15': 11, 'F-22': 11, 'B-133-new': 13}

        model_det = YOLO(self.cfg['model_detect'])
        data_pth = self.cfg['inference_data'][self.session]
        xml_pth = self.cfg['inference_ann'][self.session]
        detections = model_det(data_pth, conf=self.cfg['confidence'])

        test_num_all = 0
        pred_obj_num = 0
        gt_obj_num = 0
        LTP = 0
        gt_list = []
        img_list = []

        for idx, detection in enumerate(detections):
            image_path = detection.path
            file_name_with_extension = os.path.basename(image_path)
            file_name = os.path.splitext(file_name_with_extension)[0]
            label_gt, bbox_gt = get_xml_results(image_path, xml_pth)

            tag = 0
            for g in label_gt:
                if g in include_cls:
                    tag = 1
                    break
            if tag == 1:
                continue

            pred_boxes = detection.boxes.xywh.cpu()
            label_gt, idx_list, true_num, pred_num, LTP_per = get_coordinating_bbox(pred_boxes, bbox_gt, label_gt)
            test_num_all += max(true_num, pred_num)
            gt = [map_dict[x] for x in label_gt]
            gt_list += gt
            pred_obj_num += pred_num
            gt_obj_num += true_num
            LTP += LTP_per

            # 打开图像
            image = Image.open(image_path).convert('RGB')

            for i, idx in enumerate(idx_list):
                # 获取标注的坐标信息
                x, y, w, h = pred_boxes[i]
                # 裁剪图像
                cropped_image = image.crop(
                    ((x - 0.5 * w).item(), (y - 0.5 * w).item(), (x + 0.5 * w).item(), (y + 0.5 * h).item()))
                output_folder = data_pth + "/" + str(gt[i])
                # output_folder = data_pth + "/" + str(session) + "/" + str(gt[idx])
                os.makedirs(output_folder, exist_ok=True)
                cropped_image.save(output_folder + "/" + file_name + '_' + str(i) + ".jpg")
                img_list.append(output_folder + "/" + file_name + '_' + str(i) + ".jpg")

        evaluate_detect(pred_obj_num, gt_obj_num, LTP)

        return img_list, gt_list, pred_obj_num


def get_xml_results(image_path, xml_dir):
    name_list = []
    bbox_list = []

    file_name_with_extension = os.path.basename(image_path)
    file_name = os.path.splitext(file_name_with_extension)[0]
    xml_path = xml_dir + file_name + '.xml'
    root = ET.parse(xml_path).getroot()  # 利用ET读取xml文件
    for obj in root.iter('object'):  # 遍历所有目标框
        name = obj.find('name').text  # 获取目标框名称，即label名
        name_list.append(name)
        obj_bnd = obj.find('bndbox')
        obj_xmin = obj_bnd.find('xmin')
        obj_ymin = obj_bnd.find('ymin')
        obj_xmax = obj_bnd.find('xmax')
        obj_ymax = obj_bnd.find('ymax')
        x = (float(obj_xmin.text) + float(obj_xmax.text)) / 2
        y = (float(obj_ymin.text) + float(obj_ymax.text)) / 2
        w = float(obj_xmax.text) - float(obj_xmin.text)
        h = float(obj_ymax.text) - float(obj_ymin.text)

        bbox = torch.tensor([x, y, w, h])
        bbox_list.append(bbox)
    bbox = torch.stack(bbox_list, dim=0)
    return name_list, bbox


def get_coordinating_bbox(pred_boxes, target_boxes, label):
    true_num = target_boxes.size(0)
    pred_num = pred_boxes.size(0)
    idx_list = []
    label_list = []
    iou_list = []
    for pred_box in pred_boxes:
        max_iou_index, iou = calculate_iou(pred_box, target_boxes)
        iou_list.append(iou)
        idx_list.append(max_iou_index)
        label_list.append(label[max_iou_index])

    combined = sorted(zip(iou_list, label_list, idx_list), key=lambda x: x[0], reverse=True)
    label_list = [x[1] for x in combined[:true_num]]
    idx_list = [x[2] for x in combined[:true_num]]
    iou_list = [x[0] for x in combined[:true_num]]
    LTP = len([x for x in iou_list if x > 0.5])

    return label_list, idx_list, true_num, pred_num, LTP


def calculate_iou(pred_box, target_boxes):
    # 计算预测框和真实框的坐标
    pred_x1 = pred_box[0]
    pred_y1 = pred_box[1]
    pred_x2 = pred_box[0] + pred_box[2]
    pred_y2 = pred_box[1] + pred_box[3]

    # 计算每个真实框和预测框的交集和并集
    inter_areas = []
    for target_box in target_boxes:
        target_x1 = target_box[0]
        target_y1 = target_box[1]
        target_x2 = target_box[0] + target_box[2]
        target_y2 = target_box[1] + target_box[3]

        inter_x1 = max(pred_x1, target_x1)
        inter_y1 = max(pred_y1, target_y1)
        inter_x2 = min(pred_x2, target_x2)
        inter_y2 = min(pred_y2, target_y2)

        inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
        pred_area = pred_box[2] * pred_box[3]
        target_area = target_box[2] * target_box[3]
        union_area = pred_area + target_area - inter_area

        iou = inter_area / union_area
        inter_areas.append(iou)

    max_iou_index = torch.argmax(torch.tensor(inter_areas))
    max_iou = torch.max(torch.tensor(inter_areas))
    return max_iou_index.item(), max_iou.item()


def evaluate_detect(pred_obj_num, gt_obj_num, LTP):
    print('L-P:', LTP / pred_obj_num)
    print('L-R:', LTP / gt_obj_num)
