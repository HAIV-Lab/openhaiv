from typing import Callable

from ncdia.utils import DATASETS
from .base import BaseDataset
from ncdia.dataloader.tools import default_loader


@DATASETS.register
class MergedDataset(BaseDataset):
    """A dataset to merge multiple datasets.
    
    Args:
        datasets (list[BaseDataset]): List of datasets to merge.
        loader (callable): A function to load an image.
        transform (callable): A function/transform to apply to the image.
        target_transform (callable): A function/transform to apply to the target.
        replace_transform (bool): Whether to replace the transform.

    Examples:
        >>> dataset = MergeDataset(datasets=[dataset1, dataset2])
        >>> len(dataset)
        1000

        >>> dataset = MergedDataset()
        >>> dataset.merge(datasets=[dataset1, dataset2])
        >>> len(dataset)
        1000

    """
    def __init__(
            self,
            datasets: list[BaseDataset] = [],
            loader = default_loader,
            transform: Callable | None = None,
            target_transform: Callable | None = None,
            replace_transform: bool = False,
    ) -> None:
        super(MergedDataset, self).__init__(loader)
        self.images = []
        self.labels = []
        self.transform = transform
        self.target_transform = target_transform

        self.merge(datasets, replace_transform)
        
    def merge(
            self,
            datasets: list[BaseDataset] = [],
            replace_transform: bool = False,
    ) -> None:
        """ Merge datasets.
            Combine images and labels from multiple datasets.
            Inherit the transform and target_transform from the last dataset.
            If loader is not the same, the last dataset's loader will be used.

        Args:
            datasets (list[BaseDataset]): List of datasets to merge.
            replace_transform (bool): Whether to replace the transform.
                If True, the transform will be replaced by the last 
                dataset's transform.
        """
        for dataset in datasets:
            if "labels" not in dataset.__dict__:
                raise ValueError("Dataset should have labels.")
            labels = dataset.labels

            if "images" not in dataset.__dict__:
                raise ValueError("Dataset should have images.")
            images = dataset.images

            # Combine images and labels from multiple datasets.
            for label, image in zip(labels, images):
                self.images.append(image)
                self.labels.append(label)

            # Inherit the transform and target_transform from the last dataset.
            if replace_transform:
                if "transform" in dataset.__dict__:
                    self.transform = dataset.transform

                if "target_transform" in dataset.__dict__:
                    self.target_transform = dataset.target_transform

            # If loader is not the same, the last dataset's loader will be used.
            if "loader" in dataset.__dict__:
                self.loader = dataset.loader
    
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
        img = self.loader(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return {
            'data': img,
            'label': label,
            'attribute': [],
            'imgpath': imgpath,
        }
