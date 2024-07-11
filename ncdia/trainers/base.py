from torch.utils.data import DataLoader


class BaseTrainer(object):
    """Basic trainer class for training models.

    Args:

    
    """
    def __init__(
            self,
            max_epoch: int,
            train_loader: DataLoader,
    ):
        super(BaseTrainer, self).__init__()
        self.max_epoch = max_epoch
        self.train_loader = train_loader

    def train(self):
        """
        
        """
        self.total_iters = self.max_epoch * len(self.train_loader)