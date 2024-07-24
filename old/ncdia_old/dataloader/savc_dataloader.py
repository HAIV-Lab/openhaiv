import os
import os.path as osp
import torchvision
import numpy as np
import torch
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .utils import *
from openpyxl import load_workbook
from collections import defaultdict
from .data_util import get_transform

class remote(Dataset):

    def __init__(self, config, mode='train', index_path=None, index=None,
                 base_sess=None, crop_transform=None, secondary_transform=None):
        self.mode = mode  # [train/test/ncd]
        self.config = config
        self._pre_operate()
        self.transform = None
        self.multi_train = False  # training set or test set
        self.crop_transform = crop_transform
        self.secondary_transform = secondary_transform
       
    
        if isinstance(secondary_transform, list):
            assert (len(secondary_transform) == self.crop_transform.N_large + self.crop_transform.N_small)

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
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
        elif mode == 'test':
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
        elif mode == 'ncd':
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            # 增加一个check功能，要求检查prev和new的标签分布
            self.new_data, self.new_targets = self.SelectfromTxt(index_path)
            # prev 数据不能包括当前的测试数据
            print('Checking the ncd dataset. index: ', index)
            self.prev_data, self.prev_targets = self.SelectfromClasses(self.test_data, self.test_targets, index)
            self.data = self.new_data + self.prev_data
            self.targets = self.new_targets + self.prev_targets

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

    def _pre_operate(self):
        split_file_train = self.config.dataloader.split_file_train
        split_file_test = self.config.dataloader.split_file_test

        self.data = []
        self.targets = []
        if self.mode == 'train':
            for file_name in os.listdir(split_file_train):
                # parts = file_name.split(".")
                # label = int(parts[0])-1
                images = os.listdir(os.path.join(split_file_train, file_name))
                for k in  range(len(images)):
                    image_path = os.path.join(split_file_train, file_name,images[k])
                    self.data.append(image_path)
                    if int(file_name)>=33:
                        self.targets.append(int(file_name)-3)
                    else:
                        self.targets.append(int(file_name))
                    # self.targets.append(label)

        elif self.mode == 'test':
            for file_name in os.listdir(split_file_test):
                # parts = file_name.split(".")
                # label = int(parts[0]) - 1
                images = os.listdir(os.path.join(split_file_test, file_name))
                for k in range(len(images)):
                    image_path = os.path.join(split_file_test, file_name,images[k])
                    self.data.append(image_path)
                    if int(file_name)>=33:
                        self.targets.append(int(file_name)-3)
                    else:
                        self.targets.append(int(file_name))
                    # self.targets.append(label)
        
        elif self.mode == 'ncd':
            self.train_data = []
            self.train_targets = []
            self.test_data = []
            self.test_targets = []

            # accumulate the all train set
            for file_name in os.listdir(split_file_train):
                images = os.listdir(os.path.join(split_file_train, file_name))
                for k in range(len(images)):
                    image_path = os.path.join(split_file_train, file_name, images[k])
                    self.train_data.append(image_path)
                    if int(file_name)>=33:
                        self.train_targets.append(int(file_name)-3)
                    else:
                        self.train_targets.append(int(file_name))

            # accumulate the all test set
            for file_name in os.listdir(split_file_test):
                images = os.listdir(os.path.join(split_file_test, file_name))
                for k in range(len(images)):
                    image_path = os.path.join(split_file_test, file_name, images[k])
                    self.test_data.append(image_path)
                    if int(file_name)>=33:
                        self.test_targets.append(int(file_name)-3)
                    else:
                        self.test_targets.append(int(file_name))

    def check_element_count(self,list_data, element):
        return list_data.count(element) <= self.config.dataloader.shot

    def SelectfromTxt(self, index_path):
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = i
            parts = i.split("/")
            if len(parts) >= 3:
                extracted_string = "/".join(parts[-2:-1])
            result = self.check_element_count(targets_tmp, int(extracted_string))
            if result == True:
                data_tmp.append(img_path)
                targets_tmp.append(int(extracted_string))

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

    def sample_test_data(self, ratio):
        """
        Sample a subset of the test data for each class according to the given ratio.

        :param ratio: The fraction of data to sample from each class.
        :return: Sampled data and corresponding targets.
        """
        if self.mode != 'test':
            raise ValueError("Sampling is only applicable in test mode")

        class_indices = {}
        for idx, label in enumerate(self.targets):
            if label in class_indices:
                class_indices[label].append(idx)
            else:
                class_indices[label] = [idx]

        sampled_data = []
        sampled_targets = []
        for label, indices in class_indices.items():
            sample_size = int(len(indices) * ratio)
            sampled_indices = random.sample(indices, sample_size)
            for i in sampled_indices:
                sampled_data.append(self.data[i])
                sampled_targets.append(self.targets[i])
        self.data, self.targets = sampled_data, sampled_targets
        # return sampled_data, sampled_targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i] 
        if self.multi_train:
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
        
        sample = dict()
        sample['data'] = total_image
        sample['label'] = targets
        sample['imgpath'] = path

        return sample


def get_base_dataloader(config):
    crop_transform, secondary_transform = get_transform(config.dataloader)
    class_index = np.arange(config.dataloader.base_class)
                
    if config.dataloader.dataset == 'remote':
        trainset = remote(config=config, mode='train', index=class_index, base_sess=True,
                          crop_transform=crop_transform, secondary_transform=secondary_transform)
        testset = remote(config=config, mode='test', index=class_index)
    else:
        trainset = remote(config=config, mode='train', index=class_index, base_sess=True,
                          crop_transform=crop_transform, secondary_transform=secondary_transform)
        testset = remote(config=config, mode='test', index=class_index)

    sampler_train = None
    sampler_test = None
    if config.dataloader.num_gpus * config.dataloader.num_machines > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(trainset)
        sampler_test = torch.utils.data.distributed.DistributedSampler(testset)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.dataloader.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True, sampler=sampler_train)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=config.dataloader.test_batch_size, shuffle=False, num_workers=8, pin_memory=True, sampler=sampler_test)

    return trainset, trainloader, testloader

def get_new_dataloader(config, session):
    crop_transform, secondary_transform = get_transform(config.dataloader)
    txt_path = os.path.join(config.dataloader.txt_path, config.dataloader.dataset, "session_" + str(session + 1) + '.txt')
    class_new = np.arange(config.dataloader.base_class + session * config.dataloader.way)

    if config.dataloader.dataset == 'remote':
        trainset = remote(config=config, mode='train', index_path=txt_path,
                          crop_transform=crop_transform, secondary_transform=secondary_transform)
        testset = remote(config=config, mode='test', index=class_new)
    else:
        trainset = remote(config=config, mode='train', index_path=txt_path,
                          crop_transform=crop_transform, secondary_transform=secondary_transform)
        testset = remote(config=config, mode='test', index=class_new)

    sampler_train = None
    sampler_test = None
    if config.dataloader.num_gpus * config.dataloader.num_machines > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(trainset)
        sampler_test = torch.utils.data.distributed.DistributedSampler(testset)
        
    if config.dataloader.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=config.dataloader.num_workers, pin_memory=True, sampler=sampler_train)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.dataloader.batch_size_new, shuffle=True,
                                                  num_workers=config.dataloader.num_workers, pin_memory=True, sampler=sampler_train)
    
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=config.dataloader.test_batch_size, shuffle=False,
                                             num_workers=config.dataloader.num_workers, pin_memory=True, sampler=sampler_test)

    return trainset, trainloader, testloader

def get_ncd_dataset(config, session):
    crop_transform, secondary_transform = get_transform(config.dataloader)
    txt_path = os.path.join(config.dataloader.txt_path, config.dataloader.dataset, "session_" + str(session + 1) + '.txt')
    class_new = np.arange(config.dataloader.base_class + session * config.dataloader.way)
    class_prev = np.arange(config.dataloader.base_class + (session-1) * config.dataloader.way)

    if config.dataloader.dataset == 'remote':
        trainset = remote(config=config, mode='ncd', index_path=txt_path, index=class_prev,
                          crop_transform=crop_transform, secondary_transform=secondary_transform)
        testset = remote(config=config, mode='test', index=class_new)
    
    return trainset, testset


def get_cil_dataloader2(config, session: int): #, ncd: bool):
    """ get the base and new seesion dataloader
    args:   
        config: whole config 
        session: 0 for base dataloader, nothing different
                 > 1 (int) for new session dataloader, trainloader may vary
        ncd: A tag refers to wheher use the novel class discovery trainloader
    return:
        trainset, trainloader, testloader
    """
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(config)
    else:
        trainset, trainloader, testloader = get_new_dataloader(config, session)
    return trainset, trainloader, testloader
    