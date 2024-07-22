from torch.utils.data import DataLoader

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
    
    dataset_cfg = kwargs.pop('dataset')
    dataset = DATASETS.build(dataset_cfg)

    loader = DataLoader(dataset, **kwargs)
    return loader
