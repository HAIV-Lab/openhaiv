from ncdia_old.utils import Config

from .attr_recorder import AttrRecorder
from .base_recorder import BaseRecorder


def get_recorder(config: Config):
    recorders = {
        'base': BaseRecorder,
        'attr': AttrRecorder,
    }

    return recorders[config.recorder.name](config)
