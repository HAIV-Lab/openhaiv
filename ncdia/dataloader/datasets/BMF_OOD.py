import os
import random
from torchvision import transforms

from ncdia.utils import DATASETS
from ncdia.dataloader.tools import pil_loader
from .utils import BaseDataset


@DATASETS.register
class BMF_OOD(BaseDataset):
    """BMF dataset

    Args:
        root (str): root folder of the dataset
        split (str): split of the dataset. Should be one of 'train', 'test'.
        subset_file (str): file containing the selected images
        transform (list | str): transform to apply on the dataset.
            If str, it should be one of 'train', 'test' for predefined transforms.
    """

    # CLIP transform
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                max_size=None,
                antialias=True,
            ),
            transforms.Lambda(
                lambda image: image.convert("RGB") if image.mode != "RGB" else image
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                224,
                interpolation=transforms.InterpolationMode.BICUBIC,
                max_size=None,
                antialias=True,
            ),
            transforms.CenterCrop((224, 224)),
            transforms.Lambda(
                lambda image: image.convert("RGB") if image.mode != "RGB" else image
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    # # ResNet transform
    # train_transform = transforms.Compose([
    #     transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
    #     transforms.CenterCrop(224),
    #     transforms.RandomHorizontalFlip(0.5),
    #     transforms.RandomCrop(224, padding=4),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # test_transform = transforms.Compose([
    #     transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BILINEAR),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    def __init__(
        self,
        root: str,
        split: str = "train",
        subset_labels: list = None,
        subset_file: str = None,
        transform: list | str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split

        self.images, self.labels = self._load_data(root)

        if split == "train":
            self.transform = self.train_transform
        elif split in ["test", "val", "ood"]:
            self.transform = self.test_transform
        else:
            raise ValueError(f"Unknown transform: {transform}")

    def _load_data(self, img_dir: str):
        """Load data from root folder

        Args:
            img_dir (str): image folder of the split

        Returns:
            list: list of image paths
            list: list of labels
        """
        id_folder = "/new_data/datasets/OES/sub-dataset1-RGB-domain1/ID/test/"
        subfolders = [f.name for f in os.scandir(id_folder) if f.is_dir()]
        subfolders_sorted = sorted(subfolders, key=lambda x: x.lower())
        folder_dict = {name: idx for idx, name in enumerate(subfolders_sorted)}

        imgpaths, labels = [], []
        for index, file_name in enumerate(sorted(os.listdir(img_dir))):
            images = sorted(os.listdir(os.path.join(img_dir, file_name)))
            for k in range(len(images)):
                image_path = os.path.join(img_dir, file_name, images[k])
                imgpaths.append(image_path)
                if self.split == "test":
                    labels.append(folder_dict[file_name])
                else:
                    labels.append(index)

        return imgpaths, labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        imgpath, label = self.images[index], self.labels[index]
        img = pil_loader(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        return {
            "data": img,
            "label": label,
            "attribute": [],
            "imgpath": imgpath,
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
        with open(file, "r") as f:
            for line in f:
                img = os.path.abspath(line.strip())
                if img in images:
                    selected_images.append(img)
                    selected_labels.append(labels[images.index(img)])

        return selected_images, selected_labels
