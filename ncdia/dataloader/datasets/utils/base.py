from torch.utils.data import Dataset

from ncdia.utils import DATASETS
from ncdia.dataloader.tools import default_loader


@DATASETS.register
class BaseDataset(Dataset):
    """Base class for datasets.

    Args:
        loader (callable): A function to load an image.
    """
    def __init__(
            self,
            loader = default_loader,
    ) -> None:
        super().__init__()
        self.images = []
        self.labels = []
        self.attributes = []

        self.loader = loader

    def __len__(self) -> int:
        """Get the length of the dataset

        Returns:
            int: length of the dataset
        """
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        """Get item from dataset

        Args:
            index (int): index of the item

        Returns:
            dict: a dictionary containing the item
                - 'data': data of the item
                - 'label': label of the item
                - 'attribute': attribute of the item
                - 'imgpath': image path of the item
        """
        return {
            'data': self.images[index],
            'label': self.labels[index],
            'attribute': self.attributes[index],
            'imgpath': self.images[index],
        }
