from torch.utils.data import DataLoader

from ncdia.utils import Config

from .augmix_trainer import AugMixTrainer
from .base_trainer import BaseTrainer
from .mixup_trainer import MixupTrainer
from .savc_att_trainer import SAVCattTrainer
from .savc_trainer import SAVCTrainer
from .fact_trainer import FACTTrainer
from .alice_trainer import AliceTrainer

def get_trainer(net, train_loader: DataLoader, val_loader: DataLoader,
                config: Config):
    if type(train_loader) is DataLoader:
        trainers = {
            'base': BaseTrainer,
            'augmix': AugMixTrainer,
            'mixup': MixupTrainer,
            'savc_att': SAVCattTrainer,
            'savc': SAVCTrainer,
            'fact': FACTTrainer,
            'alice': AliceTrainer
        }
        if config.trainer.name in ['cider', 'npos']:
            return trainers[config.trainer.name](net, train_loader, val_loader,
                                                 config)
        else:
            return trainers[config.trainer.name](net, train_loader, config)