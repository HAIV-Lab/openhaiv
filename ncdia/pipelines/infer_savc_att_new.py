import numpy as np
import torch

import ncdia.utils.comm as comm
from ncdia.discoverers import get_discoverers
from ncdia.dataloader import get_dataloader
from ncdia.evaluators import get_evaluator
from ncdia.networks import get_network
from ncdia.recorders import get_recorder
from ncdia.trainers import get_trainer
from ncdia.utils import setup_logger, set_seed
from ncdia.quantize import quantize_pack, reconstruct
from ncdia.augmentations import fantasy

from ncdia.dataloader.data_util import get_transform
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Infer_SAVC_New_Pipeline:
    def __init__(self, config) -> None:
        self.config = config
        setup_logger(self.config)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        set_seed(self.config.seed)
        self.net = get_network(self.config)
        self.evaluator = get_evaluator(self.config)
        self.recorder = get_recorder(self.config)

        if self.config.CIL.fantasy is not None:
            self.transform, trans = fantasy.__dict__[self.config.CIL.fantasy]()
        else:
            self.transform, trans = None, 0
        self.crop_transform, self.secondary_transform = get_transform(config.dataloader)

        self.train_transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(224),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.test_transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def run(self, session, batch):
        test_class = self.config.CIL.base_class + session * self.config.CIL.way
        logit_list, pred_list, conf_list, label_list = [], [], [], []
        logit_att_list, pred_att_list, conf_att_list, label_att_list = [], [], [], []
        feature_list = []
        self.net.eval()
        with torch.no_grad():
            # ---------  load the data  --------  #
            imgpath = batch['imgpath']
            label = batch['label'].cuda()
            attribute = batch['attribute'].cuda()

            assert isinstance(imgpath,list), "imgpath should be list!"
            total_image = torch.stack([self.test_transform(Image.open(path).convert('RGB')) for path in imgpath])
            total_image = total_image.clone().detach()
            data = total_image.cuda()

            if data.dim() != 4:
                data = data.unsqueeze(0)
            b = data.size()[0]
            data = self.transform(data)
            m = data.size()[0] // b
            joint_preds, joint_preds_att = self.net(data)
            feat = self.net.get_features(data)
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
            conf = pred
            pred_att_list.append(pred.cpu())
            conf_att_list.append(conf.cpu())
            label_att_list.append(attribute.cpu())

        # convert values into numpy array
        feature_list = torch.cat(feature_list, dim=0).cpu()
        logit_list = torch.cat(logit_list, dim=0)
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).cpu()
        label_list = torch.cat(label_list).numpy().astype(int)
        
        logit_att_list = torch.cat(logit_att_list, dim=0)
        pred_att_list = torch.cat(pred_att_list).numpy().astype(int)
        conf_att_list = torch.cat(conf_att_list).cpu()
        label_att_list = torch.cat(label_att_list).numpy().astype(int)


        return feature_list, pred_list, logit_list, label_list, pred_att_list, logit_att_list, label_att_list