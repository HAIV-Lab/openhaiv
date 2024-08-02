from torch.utils.data import DataLoader
from torchvision.transforms import transforms

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
    dataset_cfg['transform'] = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    dataset = DATASETS.build(dataset_cfg)

    loader = DataLoader(dataset, **kwargs)
    return loader
