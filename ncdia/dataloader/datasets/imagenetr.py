import os
import random
from torchvision import transforms

from ncdia.utils import DATASETS
from ncdia.dataloader.tools import default_loader
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
        transform (torchvision.transforms.Compose): transform to apply on the dataset
    
    """
    num_classes = 1000

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
        transform: list | str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split

        