from typing import Callable

from ncdia.utils import DATASETS
from .base import BaseDataset
from ncdia.dataloader.tools import default_loader


@DATASETS.register
class MergedPairDataset(BaseDataset):
    """A dataset to merge multiple datasets.
    
    Args:
        datasets (list[BaseDataset]): List of datasets to merge.
        loader (callable): A function to load an image.
        transform (callable): A function/transform to apply to the image.
        target_transform (callable): A function/transform to apply to the target.

    Examples:
        >>> dataset = MergeDataset(datasets=[dataset1, dataset2])
        >>> len(dataset)
        1000

    """
    def __init__(
            self,
            datasets: list[BaseDataset] = [],
            loader = default_loader,
            transform: Callable | None = None,
            target_transform: Callable | None = None,
    ) -> None:
        super(MergedPairDataset, self).__init__(loader)
        self.images = {"a": [], "b": []}
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform

        self.merge(datasets)
        
    def merge(
            self,
            datasets: list[BaseDataset] = [],
            replace_transform: bool = False,
    ) -> None:
        """Merge datasets.

        Args:
            datasets (list[BaseDataset]): List of datasets to merge.
            replace_transform (bool): Whether to replace the transform.
                If True, the transform will be replaced by the last 
                dataset's transform.
        """
        for dataset in datasets:
            labels = dataset.labels
            images = dataset.images

            num_classes = self.num_classes
            label_set = list(set(labels))
            for label, image_a, image_b in zip(labels, images["a"], images["b"]):
                self.images["a"].append(image_a)
                self.images["b"].append(image_b)
                self.labels.append(
                    num_classes + label_set.index(label))

            if replace_transform:
                self.transform = dataset.transform
    
    def __len__(self) -> int:
        """Get the length of the dataset

        Returns:
            int: length of the dataset
        """
        return len(self.labels)
    
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
        imgpath_a, imgpath_b, label = self.images["a"][index], self.images["b"][index], self.labels[index]
        img_a = self.loader(imgpath_a)
        img_b = self.loader(imgpath_b)

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return {
            'data': {"a": img_a, "b": img_b},
            'label': label,
            'attribute': [],
            'imgpath': {"a": imgpath_a, "b": imgpath_b},
        }
