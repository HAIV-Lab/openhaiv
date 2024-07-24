import numpy as np
import torch
from copy import deepcopy

import ncdia_old.utils.comm as comm
from ncdia_old.discoverers import get_discoverers
from ncdia_old.dataloader import get_dataloader
from ncdia_old.evaluators import get_evaluator
from ncdia_old.networks import get_network
from ncdia_old.recorders import get_recorder
from ncdia_old.trainers import get_trainer
from ncdia_old.utils import setup_logger
from ncdia_old.quantize import quantize_train_savc, quantize_pack


class Train_SAVC_ATT_Base_Pipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # set random seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        session = 0
        # get dataloader
        get_cil_dataloader = get_dataloader(self.config)
        trainset, train_loader, val_loader = get_cil_dataloader(self.config, session)
        test_loader = val_loader

        # init network
        net = get_network(self.config)

        # init trainer and evaluator
        trainer = get_trainer(net, train_loader, val_loader, self.config)
        evaluator = get_evaluator(self.config)

        if comm.is_main_process():
            # init recorder
            recorder = get_recorder(self.config)

            print('Start training...', flush=True)

        train_loader.dataset.multi_train = True
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            # train and eval the model
            net, train_metrics = trainer.train_epoch_base(epoch_idx)
            # ######### Using attribute evaluator ######### #
            val_metrics = evaluator.eval_acc_attr(net, val_loader, None, session,
                                                 epoch_idx)
            comm.synchronize()
            if comm.is_main_process():
                # save model and report the result
                tag = recorder.save_model(net, val_metrics)
                recorder.report(train_metrics, val_metrics)
            if tag == 'best':
                best_model_dict = deepcopy(net.state_dict())

        # -------- Evaluate OOD --------- #
        net.eval()
        prev_trainset, prev_train_loader, prev_val_loader = get_cil_dataloader(self.config, session)
        cur_trainset, cur_train_loader, cur_val_loader = get_cil_dataloader(self.config, session+1)  # 3类的train
        train_tfm, test_tfm = prev_train_loader.dataset.transform, prev_val_loader.dataset.transform
        prev_train_loader.dataset.transform = test_tfm
        cur_train_loader.dataset.transform = test_tfm
        evaluator.eval_ood(net, {'train': prev_train_loader, 'test': prev_val_loader}, cur_train_loader, session)

        # -------- Replace the fc --------- #
        if not self.config.CIL.not_data_init:
            net.load_state_dict(best_model_dict)
            train_loader.dataset.multi_train = False
            train_loader.dataset.transform = val_loader.dataset.transform
            net = trainer.replace_base_fc(train_loader)
            net.mode = 'avg_cos'
            val_metrics = evaluator.eval_acc_attr(net, val_loader, None, session, epoch_idx)
            print("Updating the base fc...", flush=True)
            ######### 'refc' tag to store replace fc one #########
            val_metrics['epoch_idx'] = 'refc'
            comm.synchronize()
            if comm.is_main_process():
                # save the last epoch model (update the one)
                recorder.save_model(net, val_metrics)
                recorder.report(train_metrics, val_metrics)
            evaluator.eval_ood(net, {'train': prev_train_loader, 'test': prev_val_loader}, cur_train_loader, session)

        # -------- quantize --------- #
        if self.config.quantizer.apply:
            train_loader.dataset.transform = train_tfm
            quan_net = quantize_train_savc(self.config.quantizer, net, net.transform, train_loader)
            test_metrics = evaluator.eval_acc_attr(quan_net, test_loader, None, session)
            if comm.is_main_process():
                print('\nCompressing accuracy {:.2f}'.format( 100.0 * test_metrics['acc']),flush=True)
                test_metrics['epoch_idx'] = 'final_quantize'
                # quan_net = quantize_pack(quan_net)
                recorder.save_model(quan_net, test_metrics)
        else:
            if comm.is_main_process():
                recorder.summary()
                print(u'\u2500' * 70, flush=True)
                # evaluate on test set
                print('Start testing...', flush=True)
            test_metrics = evaluator.eval_acc_attr(net, test_loader, None, session)
            if comm.is_main_process():
                print('\nComplete Evaluation, Last accuracy {:.2f}'.format(
                    100.0 * test_metrics['acc']),
                    flush=True)
                print('Completed!', flush=True)
