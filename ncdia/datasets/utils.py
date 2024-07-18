from ncdia.utils import Config
from .get_dataloader import get_cil_dataloader2

def get_dataloader(config: Config):
    if config.dataloader.name == 'savc_att_dataloader':
        return get_cil_dataloader2
    elif config.dataloader.name == 'savc_dataloader':
        return get_cil_dataloader2
