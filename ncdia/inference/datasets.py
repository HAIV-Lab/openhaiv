from PIL import Image
import os
import xml.etree.ElementTree as ET
import torch

map_dict = {'BM-8-24': 1, 'BM-13N': 2, 'A8': 6, 'A10': 7, 'bmp2': 8, 'T62': 9, 'T34': 10, 'A15': 11, 'B133new': 13}
include_cls = ['BM-8-24', 'BM-13N', 'A8', 'A10', 'bmp2', 'T62', 'T34', 'A15', 'B133new']

def generate_classify_datasets(model_det, cfg, data_pth, xml_pth):
    detections = model_det(data_pth, conf=cfg.conf)
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
            if g not in include_cls:
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
        # for idx in range(pred_num):
        #     # 获取标注的坐标信息
        #     x, y, w, h = pred_boxes[idx]
        #     # 裁剪图像
        #     cropped_image = image.crop((x.item(), y.item(), (x + w).item(), (y + h).item()))
        #     output_folder = data_pth + "/" + str(session) + "/" + str(gt[idx])
        #     os.makedirs(output_folder, exist_ok=True)
        #     cropped_image.save(output_folder + "/" + file_name + '_' + str(idx) + ".jpg")
        #     img_list.append(output_folder + "/" + file_name + '_' + str(idx) + ".jpg")

        for idx in idx_list:
            # 获取标注的坐标信息
            x, y, w, h = pred_boxes[idx]
            # 裁剪图像
            cropped_image = image.crop(((x-0.5*w).item(), (y-0.5*w).item(), (x + 0.5*w).item(), (y + 0.5*h).item()))
            output_folder = data_pth + "/" + str(gt[idx])
            # output_folder = data_pth + "/" + str(session) + "/" + str(gt[idx])
            os.makedirs(output_folder, exist_ok=True)
            cropped_image.save(output_folder + "/" + file_name + '_' + str(idx) + ".jpg")
            img_list.append(output_folder + "/" + file_name + '_' + str(idx) + ".jpg")

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
    print('L-P:', LTP/pred_obj_num)
    print('L-R:', LTP/gt_obj_num)