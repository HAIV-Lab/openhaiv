import os
import os.path as osp
import torchvision
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import *
import tifffile as tf



class remote(Dataset):

    def __init__(self, root='./', train=True, index_path=None, index=None,
                 base_sess=None, crop_transform=None, secondary_transform=None, args=None):
        self.root = '/new_data/tyw/FSCIL/Basenew/'
        self.train = train  # training set or test set
        self._pre_operate(self.root)
        self.transform = None
        self.multi_train = False  # training set or test set
        self.crop_transform = crop_transform
        self.secondary_transform = secondary_transform
        if isinstance(secondary_transform, list):
            assert (len(secondary_transform) == self.crop_transform.N_large + self.crop_transform.N_small)

        if train:
            self.transform = transforms.Compose([
                transforms.Resize(256,interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(224),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(index_path)
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256,interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        split_file_train = self.args.split_file_train
        split_file_test = self.args.split_file_test

        self.data = []
        self.targets = []
        if self.train:
            for file_name in os.listdir(split_file_train):
                #parts = file_name.split(".")
                #label=int(parts[0])-1
                 images = os.listdir(os.path.join(split_file_train, file_name))
                 for k in  range(len(images)):
                    image_path = os.path.join(split_file_train, file_name,images[k])
                    self.data.append(image_path)
                    # if int(file_name)>=33:
                    #     self.targets.append(int(file_name)-3)
                    # else:
                    # 20240123取前3位
                    self.targets.append(int(file_name[:2]))
                    #self.targets.append(label)

        else:
            for file_name in os.listdir(  split_file_test):
                #parts = file_name.split(".")
                #label = int(parts[0]) - 1
                images = os.listdir(os.path.join(split_file_test, file_name))
                for k in range(len(images)):
                    image_path = os.path.join(split_file_test, file_name,images[k])
                    self.data.append(image_path)
                    # if int(file_name)>=33:
                    #     self.targets.append(int(file_name)-3)
                    # else:
                    # 20240123 取前3位
                    self.targets.append(int(file_name[:2]))
                    #self.targets.append(label)

    def check_element_count(self,list_data, element):
        return list_data.count(element) <= 4

    def SelectfromTxt(self, index_path):
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
                img_path = i
                parts = i.split("/")
                if len(parts) >= 3:
                    extracted_string = "/".join(parts[-2:-1])
                result = self.check_element_count(targets_tmp, int(extracted_string[:2]) )
                if result == True:
                    data_tmp.append(img_path)
                    # 只读取前两个字母作为标签
                    targets_tmp.append(int(extracted_string[:2]))

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        if self.multi_train:
            if path.endswith(".tiff"):
                # Images = tf.imread(path)
                # color_image = np.repeat(Images[:, :, np.newaxis], 3, axis=2)
                # image= Image.fromarray(color_image)
                image = Image.open(path).convert('RGB')
            else:
                image = Image.open(path).convert('RGB')
            classify_image = [self.transform(image)]
            multi_crop, multi_crop_params = self.crop_transform(image)
            assert (len(multi_crop) == self.crop_transform.N_large + self.crop_transform.N_small)
            if isinstance(self.secondary_transform, list):
                multi_crop = [tf(x) for tf, x in zip(self.secondary_transform, multi_crop)]
            else:
                multi_crop = [self.secondary_transform(x) for x in multi_crop]
            total_image = classify_image + multi_crop
        else:
            total_image = self.transform(Image.open(path).convert('RGB'))
        return path, total_image, targets