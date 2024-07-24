from ncdia.utils import Config
from .savc_att_dataloader import get_cil_dataloader
from .savc_dataloader import get_cil_dataloader2
from .fact_dataloader import get_cil_dataloader_fact
from .alice_dataloader import get_cil_dataloader_alice
from .lwf_dataloader import get_cil_dataloader_lwf

def get_dataloader(config: Config):
    if config.dataloader.name == 'savc_att_dataloader':
        return get_cil_dataloader
    elif config.dataloader.name == 'savc_dataloader':
        return get_cil_dataloader2
    elif config.dataloader.name == 'fact_dataloader':
        return get_cil_dataloader_fact
    elif config.dataloader.name == 'alice_dataloader':
        return get_cil_dataloader_alice
    elif config.dataloader.name == 'lwf_dataloader':
        return get_cil_dataloader_lwf