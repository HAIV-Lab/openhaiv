from ncdia.utils import Config

from .BiDistCosClassifier import CosClassifier

def get_model(config: Config):
    models = {
        'bidist': CosClassifier,
    }

    return pipeline[config.model.name](config)