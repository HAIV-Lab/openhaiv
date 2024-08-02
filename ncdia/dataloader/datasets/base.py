from torch.utils.data import Dataset

from ncdia.utils import DATASETS


@DATASETS.register
class BaseDataset(Dataset):
    """Base class for datasets.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.images = []
        self.labels = []
        self.attributes = []

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
