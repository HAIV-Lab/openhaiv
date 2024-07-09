import numpy as np
import torch

import ncdia.utils.comm as comm
from ncdia.discoverers import get_discoverers
from ncdia.dataloader import get_dataloader
from ncdia.evaluators import get_evaluator
from ncdia.networks import get_network
from ncdia.recorders import get_recorder
from ncdia.trainers import get_trainer
from ncdia.utils import setup_logger, set_seed

class Train_SAVC_New_Pipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # set random seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        set_seed(self.config.seed)

        # init get_discoverers
        discoverer = get_discoverers(self.config)

        # init network
        net = get_network(self.config)

        # init trainer and evaluator
        evaluator = get_evaluator(self.config)

        if comm.is_main_process():
            # init recorder
            recorder = get_recorder(self.config)

            print('Start training...', flush=True)

        for session in range(1, self.config.CIL.sessions): # start from session 1 
            # get dataloader
            get_cil_dataloader = get_dataloader(self.config)
            prev_trainset, prev_train_loader, prev_val_loader = get_cil_dataloader(self.config, session-1)
            cur_trainset, cur_train_loader, cur_val_loader = get_cil_dataloader(self.config, session)  # 3类的train
            cur_test_loader = cur_val_loader
            train_tfm, test_tfm = prev_train_loader.dataset.transform, prev_val_loader.dataset.transform
            
            trainer = get_trainer(net, cur_train_loader, cur_val_loader, self.config)
            
            # prev_train_loader.dataset.transform = test_tfm
            cur_train_loader.dataset.transform = test_tfm
            print('Testing the pretrained model', flush=True)
            val_metrics = evaluator.eval_acc(net, prev_val_loader, None, session-1, 0)
            evaluator.eval_ood(net, {'train': prev_train_loader, 'test': prev_val_loader}, cur_train_loader, session-1)

            # --------- novel class discover -------- #
            prev_val_loader.dataset.sample_test_data(ratio=self.config.discoverer.prev_valset_ratio)
            if self.config.discoverer.apply:
                cur_train_loader = discoverer.get_pseudo_newloader(net, {'train': prev_train_loader, 'test':                    prev_val_loader}, cur_train_loader, train_tfm, session-1)
            #  -------------------------------------  #
            trainer.train_loader = cur_train_loader
            
            net.mode = self.config.CIL.new_mode
            net.eval()
            # 用新的trainloader更新新的fc层
            net.update_fc(cur_train_loader, np.unique(cur_train_loader.dataset.targets), session)
            # val_metrics = evaluator.eval_acc(net, prev_val_loader, None, session-1, 0)
            val_metrics = evaluator.eval_acc(net, cur_val_loader, None, session, 0)
            # print(val_metrics)

            if self.config.CIL.incft:
                # update the optimizer and schedule for new_fc
                trainer.update_session_info(session)
                trainer.train_loader.dataset.transform = train_tfm
                trainer.train_loader.dataset.multi_train = True
                for epoch_idx in range(1, self.config.optimizer.new_epochs + 1):
                    # train and eval the model
                    net, train_metrics = trainer.train_epoch_new(epoch_idx)

                net.fc.weight.data[trainer.old_class*trainer.num_trans : trainer.new_class*trainer.num_trans, :].copy_(trainer.new_fc.data)
                val_metrics = evaluator.eval_acc(net, cur_val_loader, None, session, epoch_idx)
                comm.synchronize()
                if comm.is_main_process():
                    # save model and report the result
                    recorder.save_model(net, val_metrics)
                    recorder.report(train_metrics, val_metrics)

        if comm.is_main_process():
            recorder.summary()
            print(u'\u2500' * 70, flush=True)

            # evaluate on test set
            print('Start testing...', flush=True)

        test_metrics = evaluator.eval_acc(net, cur_test_loader, None, session)

        evaluator.cheating_testset(net, cur_test_loader, None, session)
        
        if comm.is_main_process():
            print('\nComplete Evaluation, Last accuracy {:.2f}'.format(
                100.0 * test_metrics['acc']),
                  flush=True)
            print('Completed!', flush=True)
