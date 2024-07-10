from ncdia_old.utils import Config

from .savc_att_discoverer import SAVCattDiscoverer
from .savc_discoverer import SAVCDiscoverer

def get_discoverers(config: Config):
    if config.discoverer.name == 'savc_att_discoverer' or config.discoverer.name == 'savc_sskm_discoverer':
        return SAVCattDiscoverer(config)
    elif config.discoverer.name == 'savc_discoverer':
        return SAVCDiscoverer(config)