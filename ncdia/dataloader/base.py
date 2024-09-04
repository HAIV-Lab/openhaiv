from torch.utils.data import DataLoader
from .augmentations import build_transform

from ncdia.utils import DATASETS


def build_dataloader(kwargs):
    """Build data loader.

    Args:
        kwargs (dict): Arguments for DataLoader. Contains the following:
            - dataset (dict): Dataset configuration.
            - other arguments for DataLoader, such as `batch_size`, `shuffle`, etc.
    
    Returns:
        loader (DataLoader): Data loader.

    Raises:
        ValueError: Dataset configuration not provided.
    """
    if 'dataset' not in kwargs:
        raise ValueError("Dataset configuration not provided")
    
    dataset_cfg = dict(kwargs.pop('dataset'))
    dataset_cfg['transform'] = build_transform(dataset_cfg['transform'])
    dataset = DATASETS.build(dataset_cfg)

    loader = DataLoader(dataset, **kwargs)
    return loader
