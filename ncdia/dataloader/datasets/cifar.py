from typing import Callable
from torchvision.datasets import CIFAR10 as _CIFAR10
from torchvision.datasets import CIFAR100 as _CIFAR100
from PIL import Image

from ncdia.utils import DATASETS
from .utils import BaseDataset


@DATASETS.register
class CIFAR10(_CIFAR10, BaseDataset):
    """CIFAR-10 dataset.

    Args:
        root (str): Root directory of dataset, e.g., "/datasets/cifar10/"
        train (bool): If True, creates dataset from training set, 
            otherwise creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image 
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the 
            target and transforms it.
        download (bool): If True, downloads the dataset from the internet and puts it 
            in root directory. If dataset is already downloaded, it is not downloaded again.

    Examples:
        >>> from ncdia.datasets import CIFAR10
        >>> dataset = CIFAR10(root='/datasets/cifar10/', train=True)
        >>> print(len(dataset))
        50000
        >>> batch = dataset[0]
        >>> print(batch['data'].size, batch['label'])
        (3, 32, 32) 6
    
    """
    num_classes = 10

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Callable | None = None,
            target_transform: Callable | None = None,
            download: bool = False,
            loader = Image.fromarray,
            **kwargs,
    ) -> None:
        _CIFAR10.__init__(
            self, root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.images, self.labels = [], []
        for path, target in zip(self.data, self.targets):
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
                - 'target': class_index of the target class.
                - 'attribute': attribute of the image,
                - 'imgpath': path of the image.
        """
        img, target = self.images[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {
            'data': img,
            'label': target,
            'attribute': [],
            'imgpath': '',
        }


@DATASETS.register
class CIFAR100(CIFAR10, _CIFAR100):
    """CIFAR-100 dataset.

    Args:
        root (str): Root directory of dataset, e.g., "/datasets/cifar100/"
        train (bool): If True, creates dataset from training set, 
            otherwise creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image 
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the 
            target and transforms it.
        download (bool): If True, downloads the dataset from the internet and puts it 
            in root directory. If dataset is already downloaded, it is not downloaded again.

    Examples:
        >>> from ncdia.datasets import CIFAR100
        >>> dataset = CIFAR100(root='/datasets/cifar100/', train=True)
        >>> print(len(dataset))
        50000
        >>> batch = dataset[0]
        >>> print(batch['data'].size, batch['label'])
        (3, 32, 32) 19

    """
    num_classes = 100
