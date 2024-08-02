from ncdia.utils import Configs
from .get_dataloader import get_cil_dataloader2

def get_dataloader(config: Configs):
    if config.dataloader.name == 'remote_att_dataloader':
        return get_cil_dataloader2
    elif config.dataloader.name == 'remote_dataloader':
        return get_cil_dataloader2
