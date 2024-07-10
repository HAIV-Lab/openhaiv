from ncdia_old.utils import Config
from .savc_att_dataloader import get_cil_dataloader
from .savc_dataloader import get_cil_dataloader2


def get_dataloader(config: Config):
    if config.dataloader.name == 'savc_att_dataloader':
        return get_cil_dataloader
    elif config.dataloader.name == 'savc_dataloader':
        return get_cil_dataloader2