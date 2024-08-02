import numpy as np
import torch
from .data_manager import *


# if config.dataloader.dataset == 'remote':
def get_base_dataloader(config, datamgr: DataManager):
    class_index = np.arange(config.dataloader.base_class)
    trainset = datamgr.get_dataset(class_index, "train", "train")
    testset = datamgr.get_dataset(class_index, "test", "test")

    sampler_train = None
    sampler_test = None
    if config.dataloader.num_gpus * config.dataloader.num_machines > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(trainset)
        sampler_test = torch.utils.data.distributed.DistributedSampler(testset)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.dataloader.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True, sampler=sampler_train)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=config.dataloader.test_batch_size, shuffle=False, 
                                             num_workers=8, pin_memory=True, sampler=sampler_test)

    return trainset, trainloader, testloader


# if config.dataloader.dataset == 'remote':
def get_new_dataloader(config, datamgr: DataManager, session: int):
    # txt_path = os.path.join(config.dataloader.txt_path, config.dataloader.dataset, "session_" + str(session + 1) + '.txt')
    class_new = np.arange(config.dataloader.base_class + session * config.dataloader.way)
    trainset = datamgr.get_dataset(class_new, "train", "train")
    testset = datamgr.get_dataset(class_new, "test", "test")

    sampler_train = None
    sampler_test = None
    if config.dataloader.num_gpus * config.dataloader.num_machines > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(trainset)
        sampler_test = torch.utils.data.distributed.DistributedSampler(testset)
        
    if config.dataloader.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=config.dataloader.num_workers, pin_memory=True, sampler=sampler_train)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.dataloader.batch_size_new, shuffle=True,
                                                  num_workers=config.dataloader.num_workers, pin_memory=True, sampler=sampler_train)
    
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=config.dataloader.test_batch_size, shuffle=False,
                                             num_workers=config.dataloader.num_workers, pin_memory=True, sampler=sampler_test)

    return trainset, trainloader, testloader


# if config.dataloader.dataset == 'remote':
def get_ncd_dataset(config, datamgr: DataManager, session):
    # txt_path = os.path.join(config.dataloader.txt_path, config.dataloader.dataset, "session_" + str(session + 1) + '.txt')
    class_new = np.arange(config.dataloader.base_class + session * config.dataloader.way)
    class_prev = np.arange(config.dataloader.base_class + (session-1) * config.dataloader.way)

    trainset = datamgr.get_dataset(class_prev, 'ncd', 'ncd')
    testset = datamgr.get_dataset(class_new, 'ncd', 'ncd')
    
    return trainset, testset


def get_cil_dataloader(config, datamgr: DataManager, session: int): #, ncd: bool):
    """ get the base and new seesion dataloader
    args:   
        config: whole config 
        session: 0 for base dataloader, nothing different
                 > 1 (int) for new session dataloader, trainloader may vary
        ncd: A tag refers to wheher use the novel class discovery trainloader
    return:
        trainset, trainloader, testloader
    """
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(config, datamgr)
    else:
        trainset, trainloader, testloader = get_new_dataloader(config, datamgr, session)
    return trainset, trainloader, testloader
