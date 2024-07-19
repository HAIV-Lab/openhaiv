import os
from typing import Any, Tuple

from torchvision.datasets import ImageFolder
from ncdia.utils import DATASETS


@DATASETS.register()
class ImageNet(ImageFolder):
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
        >>> img, target, attribute, imgpath = dataset[0]
        >>> print(img.size, target)
        (3, 224, 224) 0
    
    """
    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: callable | None = None,
            target_transform: callable | None = None,
    ):
        assert split in ['train', 'val'], f"Unsupported split: {split}"
        root = os.path.join(root, split)

        super(ImageNet, self).__init__(
            root,
            transform,
            target_transform,
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            batch (tuple): (image, target, attribute, imgpath), where 
                target is class_index of the target class,
                attribute is the attribute of the image,
                imgpath is the path of the image.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, None, path
