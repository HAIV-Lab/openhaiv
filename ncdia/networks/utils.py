from copy import deepcopy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import ncdia.utils.comm as comm
from ncdia.quantize import quantize_pack, reconstruct

from .resnet18_savc_att import SAVCNET
from .resnet18_savc_att_q import SAVCNET_q
from .resnet18_savc_q import SAVCNET_q2
from .resnet18_savc_q_sar import SAVCNET_q2_sar
from .resnet18_savc_att_q_ir import SAVCNET_q_ir
from .resnet18_fact import FACTNET
from .resnet18_alice import AliceNET


def get_network(config):
    network_config = config.network
    num_classes = network_config.num_classes

    if network_config.name == 'resnet18_savc_att':
        net = SAVCNET(args=config)
    elif network_config.name == 'resnet18_savc_att_q':
        net = SAVCNET_q(args=config)
    elif network_config.name == 'resnet18_savc_q':
        net = SAVCNET_q2(args=config)
    elif network_config.name == 'resnet18_savc_q_sar':
        net = SAVCNET_q2_sar(args=config)
    elif network_config.name == 'resnet18_savc_att_q_ir':
        net = SAVCNET_q_ir(args=config)
    elif network_config.name == 'resnet18_fact':
        net = FACTNET(args=config)
    elif network_config.name == 'resnet18_alice':
        net = AliceNET(args=config)
    else:
        raise Exception('Unexpected Network Architecture!')

    if network_config.pretrained:
        print('Using pretrained model')

        # -------- quantize --------- #
        if config.quantizer.apply:
            print('!!!!!!! Loading Compressing Models !!!!!!!', flush=True)
            net = reconstruct(net, config.quantizer.quant)
        # -------- -------- --------- #

        if type(net) is dict:
            if isinstance(network_config.checkpoint, list):
                for subnet, checkpoint in zip(net.values(),
                                              network_config.checkpoint):
                    if checkpoint is not None:
                        if checkpoint != 'none':
                            subnet.load_state_dict(torch.load(checkpoint),
                                                   strict=False)
            elif isinstance(network_config.checkpoint, str):
                ckpt = torch.load(network_config.checkpoint)
                subnet_ckpts = {k: {} for k in net.keys()}
                for k, v in ckpt.items():
                    for subnet_name in net.keys():
                        if k.startwith(subnet_name):
                            subnet_ckpts[subnet_name][k.replace(
                                subnet_name + '.', '')] = v
                            break

                for subnet_name, subnet in net.items():
                    subnet.load_state_dict(subnet_ckpts[subnet_name])

        elif network_config.name == 'bit' and not network_config.normal_load:
            net.load_from(np.load(network_config.checkpoint))
        elif network_config.name == 'vit':
            pass
        else:
            try:
                net.load_state_dict(torch.load(network_config.checkpoint),
                                    strict=False)
            except RuntimeError:
                # sometimes fc should not be loaded
                loaded_pth = torch.load(network_config.checkpoint)
                loaded_pth.pop('fc.weight')
                loaded_pth.pop('fc.bias')
                net.load_state_dict(loaded_pth, strict=False)
        print('Model Loading {} Completed!'.format(network_config.name))

    if network_config.num_gpus > 1:
        if type(net) is dict:
            for key, subnet in zip(net.keys(), net.values()):
                net[key] = torch.nn.parallel.DistributedDataParallel(
                    subnet.cuda(),
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=True)
        else:
            net = torch.nn.parallel.DistributedDataParallel(
                net.cuda(),
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=True)

    if network_config.num_gpus > 0:
        if type(net) is dict:
            for subnet in net.values():
                subnet.cuda()
        else:
            net.cuda()

    cudnn.benchmark = True
    return net
