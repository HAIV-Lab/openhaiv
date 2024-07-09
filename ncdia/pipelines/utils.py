from ncdia.utils import Config

from .train_savc_att_base import Train_SAVC_ATT_Base_Pipeline
from .train_savc_att_new import Train_SAVC_ATT_New_Pipeline
from .train_savc_base import Train_SAVC_Base_Pipeline
from .train_savc_new import Train_SAVC_New_Pipeline
from .infer_savc_att_new import Infer_SAVC_New_Pipeline

def get_pipeline(config: Config):
    pipelines = {
        'savc_att_base': Train_SAVC_ATT_Base_Pipeline,
        'savc_att_new': Train_SAVC_ATT_New_Pipeline,
        'savc_base': Train_SAVC_Base_Pipeline,
        'savc_new': Train_SAVC_New_Pipeline,
        'savc_att_infer': Infer_SAVC_New_Pipeline,
    }

    return pipelines[config.pipeline.name](config)


