import os
import random
from torchvision import transforms

from ncdia.utils import DATASETS
from ncdia.dataloader.tools import pil_loader
from .utils import BaseDataset


@DATASETS.register
class ImageNetR(BaseDataset):
    """ImageNetR dataset

    Args:
        root (str): root folder of the dataset
        split (str): split of the dataset. Should be one of 'train', 'test'.
        subset_labels (list | int): subset of labels
        subset_file (str): file containing the selected images
        transform (list | str): transform to apply on the dataset.
            If str, it should be one of 'train', 'test' for predefined transforms.

    Attributes:
        images (list): list of image paths
        labels (list): list of labels
        num_classes (int): number of classes
        transform (torchvision.transforms.Compose): transform to apply on the dataset
    
    """
    train_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        subset_labels: list = None,
        transform: list | str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split

        self.images, self.labels = self._load_data()

        if subset_labels is not None:
            self.images, self.labels = self._select_from_label(self.images, self.labels, subset_labels)

        if isinstance(transform, str):
            if transform == 'train':
                self.transform = self.train_transform
            elif transform == 'test':
                self.transform = self.test_transform
            else:
                raise ValueError(f"Unknown transform: {transform}")
        else:
            self.transform = transform
    
    def _load_data(self):
        data= []
        targets = []
        label_cnt = -1

        if self.split == 'train':
            for subdir in os.listdir(os.path.join(self.root, 'train')):
                subdir_path = os.path.join(self.root, 'train', subdir)
                if os.path.isdir(subdir_path):
                    label_cnt +=1
                    for file_name in os.listdir(subdir_path):
                        img_path = os.path.join(subdir_path, file_name)
                        data.append(img_path)
                        targets.append(label_cnt)
        else:
            for subdir in os.listdir(os.path.join(self.root, 'test')):
                subdir_path = os.path.join(self.root, 'test', subdir)
                if os.path.isdir(subdir_path):
                    label_cnt +=1
                    for file_name in os.listdir(subdir_path):
                        img_path = os.path.join(subdir_path, file_name)
                        data.append(img_path)
                        targets.append(label_cnt)
        
        return data, targets

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int)-> dict:
        imgpath, label = self.images[index], self.labels[index]
        img = pil_loader(imgpath)

        if self.transform is not None:
            img = self.transform(img)
        
        return {
            'data': img,
            'label': label,
            'attribute': [],
            'imgpath': imgpath,
        }
    
    def _select_from_label(self, images: list, labels: list, subset_labels: list | int):
        """Select images from a subset of labels

        Args:
            images (list): list of image paths
            labels (list): list of labels
            subset_labels (list | int): list of subset labels

        Returns:
            list: list of image paths
            list: list of labels
        """
        if isinstance(subset_labels, int):
            subset_labels = [i for i in range(subset_labels)]
        selected_images, selected_labels = [], []
        for img, label in zip(images, labels):
            if label in subset_labels:
                selected_images.append(img)
                selected_labels.append(label)
        return selected_images, selected_labels
        