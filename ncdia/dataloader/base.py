from copy import deepcopy
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
        _dataset_kwargs (dict): Dataset configuration.
        _loader_kwargs (dict): DataLoader configuration.

    Raises:
        ValueError: Dataset configuration not provided.

    """
    if "dataset" not in kwargs:
        raise ValueError("Dataset configuration not provided")

    dataset_cfg = dict(kwargs.pop("dataset"))
    _dataset_kwargs = deepcopy(dataset_cfg)

    if dataset_cfg["transform"]:
        dataset_cfg["transform"] = build_transform(dataset_cfg["transform"])

    dataset = DATASETS.build(dataset_cfg)

    loader = DataLoader(dataset, **kwargs)
    _loader_kwargs = deepcopy(kwargs)

    return loader, _dataset_kwargs, _loader_kwargs
