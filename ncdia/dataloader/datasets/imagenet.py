import os
from typing import Callable
from torchvision.datasets import ImageFolder

from ncdia.utils import DATASETS
from .utils import BaseDataset
from ncdia.dataloader import default_loader


@DATASETS.register
class ImageNet(ImageFolder, BaseDataset):
    """ImageNet dataset.

    Args:
        root (str): Root directory of dataset, e.g., "/datasets/imagenet/"
        split (str): The dataset split, supports 'train' and 'val'.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    Examples:
        >>> from ncdia.datasets import ImageNet
        >>> dataset = ImageNet(root='/datasets/imagenet/', split='train')
        >>> print(len(dataset))
        1281167
        >>> batch = dataset[0]
        >>> print(batch['data'].size, batch['label'])
        (3, 224, 224) 0
    
    """
    num_classes = 1000

    def __init__(
            self,
            root: str,
            split: str = "train",
            loader = default_loader,
            transform: Callable | None = None,
            target_transform: Callable | None = None,
            **kwargs,
    ) -> None:
        assert split in ['train', 'val'], f"Unsupported split: {split}"
        root = os.path.join(root, split)

        ImageFolder.__init__(self, root, transform, target_transform, loader)

        self.images, self.labels = [], []
        for path, target in self.samples:
            self.images.append(path)
            self.labels.append(target)
            
        self.loader = loader
    
    def __len__(self) -> int:
        """Get the length of the dataset

        Returns:
            int: length of the dataset
        """
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        """
        Args:
            index (int): Index

        Returns:
            batch (dict):
                - 'data': image data,
                - 'target': class_index of the target class,
                - 'attribute': attribute of the image,
                - 'imgpath': path of the image.
        """
        imgpath, label = self.images[index], self.labels[index]
        sample = self.loader(imgpath)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return {
            'data': sample,
            'label': label,
            'attribute': [],
            'imgpath': imgpath,
        }
