import os
import random
from torchvision import transforms

from ncdia.utils import DATASETS
from ncdia.dataloader.tools import pil_loader
from .base import BaseDataset

@DATASETS.register
class Caltech101(BaseDataset):
    """
    Caltech101 dataset
    Args:
        root (str): root folder of the dataset
        split (str): split of the dataset. Should be one of 'train', 'test'.
        subset_file (str): file containing the selected images
        transform (list | str): transform to apply on the dataset.
            If str, it should be one of 'train', 'test' for predefined transforms.
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
        subset_file: str = None,
        transform: list | str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split

        if split == 'train':
            self.images, self.labels = self._load_data(os.path.join(root, 'train'))
        elif split == 'test':
            self.images, self.labels = self._load_data(os.path.join(root, 'test'))
        else:
            raise ValueError(f"Unknown split: {split}")

        if subset_labels is not None:
            self.images, self.labels = self._select_from_label(self.images, self.labels, subset_labels)
        if subset_file is not None:
            self.images, self.labels = self._select_from_file(self.images, self.labels, subset_file)


        if isinstance(transform, str):
            if transform == 'train':
                self.transform = self.train_transform
            elif transform == 'test':
                self.transform = self.test_transform
            else:
                raise ValueError(f"Unknown transform: {transform}")
        else:
            self.transform = transform
        

    def _load_data(self, root:str):
        """Load data from root folder

        Args:
            root (str): root folder of the dataset

        Returns:
            list: list of image paths
            list: list of labels
        """
        imgpaths, labels = [], []
        if self.split == 'train':
            split_file_path = os.path.join(root, 'train')
        else:
            split_file_path = os.path.join(root, 'test')
        imgpaths = []
        labels = []
        for file_name in os.listdir(split_file_path):
            images = os.listdir(os.path.join(split_file_path, file_name))
            for k in  range(len(images)):
                image_path = os.path.join(split_file_path, file_name,images[k])
                imgpaths.append(image_path)
                labels.append(int(file_name))
        return imgpaths, labels
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
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

    def _select_from_file(self, images: list, labels: list, file: str):
        """Select images from a subset of labels

        Args:
            images (list): list of image paths
            labels (list): list of labels
            file (str): file containing the selected images

        Returns:
            list: list of image paths
            list: list of labels
        """
        selected_images, selected_labels = [], []
        with open(file, 'r') as f:
            for line in f:
                img = os.path.abspath(line.strip())
                if img in images:
                    selected_images.append(img)
                    selected_labels.append(labels[images.index(img)])
        return selected_images, selected_labels