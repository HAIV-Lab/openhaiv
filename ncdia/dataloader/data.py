import os
import numpy as np
from torchvision import datasets, transforms
from .registry import DatasetRegister, iData
DATASETS = DatasetRegister()


def split_images_labels(pairs: list):
    # split trainset.imgs in ImageFolder
    images, labels = zip(*pairs)
    return np.array(list(images)), np.array(list(labels))


@DATASETS.register_module()
class iCIFAR10(iData):
    def __init__(self, path: str = "./data") -> None:
        super().__init__(path)
        self.use_path = False
        self.train_trsf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
        ]
        self.test_trsf = [transforms.ToTensor()]
        self.common_trsf = [
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            ),
        ]

        self.class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(self.path, train=True, download=True) # "./data"
        test_dataset = datasets.cifar.CIFAR10(self.path, train=False, download=True) # "./data"
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


@DATASETS.register_module()
class iCIFAR100(iData):
    def __init__(self, path: str = "./data") -> None:
        super().__init__(path)
        self.use_path = False
        self.train_trsf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor()
        ]
        self.test_trsf = [transforms.ToTensor()]
        self.common_trsf = [
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
            ),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(self.path, train=True, download=True) # "./data"
        test_dataset = datasets.cifar.CIFAR100(self.path, train=False, download=True) # "./data"
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


@DATASETS.register_module()
class iImageNet1000(iData):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.use_path = True
        self.train_trsf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
        ]
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.class_order = np.arange(1000).tolist()

    def download_data(self):
        if os.path.exists(self.path):
            train_dir = os.path.join(self.path, "train")
            test_dir = os.path.join(self.path, "val")

            train_dset = datasets.ImageFolder(train_dir)
            test_dset = datasets.ImageFolder(test_dir)

            self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
            self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        else:
            assert 0, "You should specify the folder of your dataset"


@DATASETS.register_module()
class iImageNet100(iData):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.use_path = True
        self.train_trsf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.class_order = np.arange(1000).tolist()

    def download_data(self):
        if os.path.exists(self.path):
            train_dir = os.path.join(self.path, "train")
            test_dir = os.path.join(self.path, "val")

            train_dset = datasets.ImageFolder(train_dir)
            test_dset = datasets.ImageFolder(test_dir)

            self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
            self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        else:
            assert 0, "You should specify the folder of your dataset"


@DATASETS.register_module()
class Remote(iData):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.use_path = True
        self.train_trsf = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
        ]
        self.ncd_trsf = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.class_order = np.arange(14).tolist()

    def download_data(self):
        if os.path.exists(self.path):
            train_dir = os.path.join(self.path, "train_0422")
            test_dir = os.path.join(self.path, "test_0422")

            train_dset = datasets.ImageFolder(train_dir)
            test_dset = datasets.ImageFolder(test_dir)

            self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
            self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

            print("Num of train: {}".format(len(self.train_targets)))
            print("Num of test: {}".format(len(self.test_targets)))
        else:
            raise DataError(self.__class__.__name__)


@DATASETS.register_module()
class Remote_detect_test(iData):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.use_path = True
        self.train_trsf = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
        ]
        self.ncd_trsf = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.class_order = np.arange(14).tolist()

    def download_data(self):
        if os.path.exists(self.path):
            train_dir = os.path.join(self.path, "train_0422")
            test_dir = os.path.join(self.path, "test_0422")

            train_dset = datasets.ImageFolder(train_dir)
            test_dset = datasets.ImageFolder(test_dir)

            self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
            self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

            print("Num of train: {}".format(len(self.train_targets)))
            print("Num of test: {}".format(len(self.test_targets)))
        else:
            raise DataError(self.__class__.__name__)


class DataError(Exception):
    def __init__(self, classname: str, *args: object) -> None:
        super().__init__(*args)
        self.msg = "Data Error, you don't have the access to {} dataset!".format(classname)

    def __str__(self) -> str:
        return self.msg
    

if __name__ == "__main__":
    cil_path = "/new_data/cyf/CIL_Dataset"
    # remote = Remote(cil_path)
    # remote.download_data()
    print(DATASETS)
