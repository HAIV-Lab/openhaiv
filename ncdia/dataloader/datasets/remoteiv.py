import os
import random
from torchvision import transforms

from ncdia.utils import DATASETS
from ncdia.dataloader.tools import pil_loader
from .utils import BaseDataset


@DATASETS.register
class Remoteiv(BaseDataset):
    """Remoteiv dataset

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
            root_a: str,
            root_b: str,
            split: str = 'train',
            subset_labels: list = None,
            subset_file: str = None,
            transform: list | str = None,
            **kwargs,
    ) -> None:
        super().__init__()
        self.root_a = root_a
        self.root_b = root_b
        self.split = split

        if split == 'train':
            self.images, self.labels = self._load_data(os.path.join(self.root_a, 'train'), \
                                                        os.path.join(self.root_b, 'train'))
        elif split == 'test':
            self.images, self.labels = self._load_data(os.path.join(self.root_a, 'test'), \
                                                        os.path.join(self.root_b, 'test'))
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

    def _load_data(self, root_a: str, root_b: str):
        """Load data from root folder

        Args:
            root (str): root folder of the dataset

        Returns:
            list: list of image paths
            list: list of labels
        """
        imgpaths, labels = {"a": [], "b": []}, []
        for dir_name in os.listdir(root_a):
            img_list_a = os.listdir(os.path.join(root_a, dir_name))
            img_list_b = os.listdir(os.path.join(root_b, dir_name))
            
            mix_img_list = list(set(img_list_a) & set(img_list_b))

            for img_name in mix_img_list:
                imgpath_a = os.path.join(root_a, dir_name, img_name)
                imgpath_b = os.path.join(root_b, dir_name, img_name)

                imgpaths["a"].append(os.path.abspath(imgpath_a))
                imgpaths["b"].append(os.path.abspath(imgpath_b))
                if int(dir_name) >= 33:
                    labels.append(int(dir_name) - 3)
                else:
                    labels.append(int(dir_name))
        return imgpaths, labels
    
    def _select_from_label(self, images: dict, labels: list, subset_labels: list | int):
        """Select images from a subset of labels

        Args:
            images (dict): dict of infrared and visble image paths
            labels (list): list of labels
            subset_labels (list | int): list of subset labels

        Returns:
            list: list of image paths
            list: list of labels
        """
        if isinstance(subset_labels, int):
            subset_labels = [i for i in range(subset_labels)]
        selected_images, selected_labels = {"a":[], "b":[]}, []
        for img_a, img_b, label in zip(images["a"], images["b"], labels):
            if label in subset_labels:
                selected_images["a"].append(img_a)
                selected_images["b"].append(img_b)
                selected_labels.append(label)
        return selected_images, selected_labels

    def _select_from_file(self, images: dict, labels: list, file: str):
        """Select images from a subset of labels

        Args:
            images (dict): dict of infrared and visble image paths
            labels (list): list of labels
            file (str): file containing the selected images

        Returns:
            list: list of image paths
            list: list of labels
        """
        selected_images, selected_labels = {"a": [], "b": []}, []
        with open(file, 'r') as f:
            for line in f:
                img = os.path.abspath(line.strip())
                if img in images["a"] and img not in selected_images["a"]:
                    selected_images["a"].append(img)
                    selected_images["b"].append(images["b"][images["a"].index(img)])
                    selected_labels.append(labels[images["a"].index(img)])
                
                if img in images["b"] and img not in selected_images["b"]:
                    selected_images["b"].append(img)
                    selected_images["a"].append(images["a"][images["b"].index(img)])
                    selected_labels.append(labels[images["b"].index(img)])

        return selected_images, selected_labels
    
    def sample_test_data(self, ratio: int | float):
        """Sample a subset of the test data for each class according to the given ratio.

        Args:
            ratio (int | float): ratio of the sampled data.
                If int, it represents the number of samples to be selected.
                If float, it represents the ratio of the samples to be selected.
        
        Returns:
            Sampled data and corresponding targets.
        """
        if self.split != 'test':
            raise ValueError("Sampling is only applicable in test mode")

        class_indices = {}
        for idx, label in enumerate(self.labels):
            if label in class_indices:
                class_indices[label].append(idx)
            else:
                class_indices[label] = [idx]

        sampled_images, sampled_labels = {"a": [], "b": []}, []
        for label, indices in class_indices.items():
            if isinstance(ratio, int) and ratio > 0:
                sample_size = min(len(indices), ratio)
            elif isinstance(ratio, float) and 0 < ratio < 1:
                sample_size = max(1, int(len(indices) * ratio))
            else:
                raise ValueError("Invalid ratio value")
            
            sampled_indices = random.sample(indices, sample_size)
            for i in sampled_indices:
                sampled_images["a"].append(self.images["a"][i])
                sampled_images["b"].append(self.images["b"][i])
                sampled_labels.append(self.labels[i])

        self.images, self.labels = sampled_images, sampled_labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        imgpath_a, imgpath_b, label = self.images["a"][index], self.images["b"][index], self.labels[index]
        img_a = pil_loader(imgpath_a)
        img_b = pil_loader(imgpath_b)

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        return {
            'data': {"a": img_a, "b": img_b},
            'label': label,
            'attribute': [],
            'imgpath': {"a": imgpath_a, "b": imgpath_b},
        }

