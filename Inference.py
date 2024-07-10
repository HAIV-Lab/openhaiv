from ncdia_old.pipelines import get_pipeline
from ncdia_old.utils import launch, set_seed 
from ncdia_old.utils.config import parse_config, Config, merge_configs, consume_dots, traverse_dfs, init_assign
import os
import argparse

# Choose one of the configurations
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch

def setup_config(config_process_order=('merge', 'parse_args', 'parse_refs')):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default=[
                "/new_data/dx450/Project/zzh_ood/NCD_Attr_zzh/configs/increment/savc_remote.yml",  # CIL参数的yml
                "/new_data/dx450/Project/zzh_ood/NCD_Attr_zzh/configs/dataloader/savc_att_remote.yml",  # 数据集的yml
                "/new_data/dx450/Project/zzh_ood/NCD_Attr_zzh/configs/networks/resnet18_savc_att.yml",  # net设置的yml
                "/new_data/dx450/Project/zzh_ood/NCD_Attr_zzh/configs/pipelines/infer_savc_att.yml"  # 修改ckpt的yml
                ])
    opt, unknown_args = parser.parse_known_args()
    config = [Config(path) for path in opt.config]
    for process in config_process_order:
        if process == 'merge':
            config = merge_configs(*config)
        elif process == 'parse_args':
            if isinstance(config, Config):
                config.parse_args(unknown_args)
            else:
                for cfg in config:
                    cfg.parse_args(unknown_args)
        elif process == 'parse_refs':
            if isinstance(config, Config):
                config.parse_refs()
            else:
                for cfg in config:
                    cfg.parse_refs()
        else:
            raise ValueError('unknown config process name: {}'.format(process))
    config.output_dir = os.path.join(config.output_dir, config.exp_name)
    return config

if __name__ == '__main__':

    config = setup_config()
    torch.use_deterministic_algorithms(True)
    set_seed(config.seed)

    prototype = torch.load(os.path.join(os.path.dirname(config.network.checkpoint), 'prototype.pth'))

    session = 1  #需要指定session，用于指定分类器fc层的维度
    batch = {'imgpath': ['/new_data/dx450/Project/zzh_ood/NCD_Attr_zzh/data/Benckmark/test_cheat/01/0011.jpg',
                         '/new_data/dx450/Project/zzh_ood/NCD_Attr_zzh/data/Benckmark/test_cheat/01/0011.jpg'],
             'label': torch.tensor([1,1]), 
             'attribute': torch.tensor([[0]*86, [0]*86])}
    
    pipeline = get_pipeline(config)
    feature_list, pred_list, logit_list, label_list, pred_att_list, logit_att_list, label_att_list = \
            pipeline.run(session, batch)
    
    # print可以将print内容保存到log文件
    print('\n\033[1;33m  Inference Done \033[0m')
