import os
import time
from pathlib import Path

import torch


class AttrRecorder:
    def __init__(self, config) -> None:
        self.config = config

        self.best_acc = 0.0
        self.best_epoch_idx = 0

        self.begin_time = time.time()
        self.output_dir = config.output_dir

    def report(self, train_metrics, val_metrics):
        print('\n\033[1;33m Epoch {:03d} | Time {:5d}s | Train Loss {:.4f}| Val Acc {:.2f}| Val mAP {:.2f} \n \
              | OP {:.2f}| OR {:.2f}| OF1 {:.2f}| CP {:.2f}| CR {:.2f}| CF1 {:.2f}\033[0m'.format(
                  (train_metrics['epoch_idx']),int(time.time() - self.begin_time), train_metrics['loss'],100.0 * val_metrics['acc'], 100.0 *val_metrics['mAP'],
                  val_metrics['OP'], val_metrics['OR'], val_metrics['OF1'], val_metrics['CP'], val_metrics['CR'], val_metrics['CF1'],
                  ),
              flush=True)

    def save_model(self, net, val_metrics):
        try:
            state_dict = net.module.state_dict()
        except AttributeError:
            state_dict = net.state_dict()
        tag = 'common'
        if self.config.recorder.save_all_models:
            torch.save(
                state_dict,
                os.path.join(
                    self.output_dir,
                    'model_epoch{}.ckpt'.format(val_metrics['epoch_idx'])))

        # enter only if better accuracy occurs
        if val_metrics['acc'] >= self.best_acc:
            # delete the depreciated best model
            old_fname = 'best_epoch{}_acc{:.4f}.ckpt'.format(
                self.best_epoch_idx, self.best_acc)
            old_pth = os.path.join(self.output_dir, old_fname)
            Path(old_pth).unlink(missing_ok=True)

            # update the best model
            self.best_epoch_idx = val_metrics['epoch_idx']
            self.best_acc = val_metrics['acc']
            torch.save(state_dict, os.path.join(self.output_dir, 'best.ckpt'))

            save_fname = 'best_epoch{}_acc{:.4f}.ckpt'.format(
                self.best_epoch_idx, self.best_acc)
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(state_dict, save_pth)
            tag = 'best'

        # save last path
        if val_metrics['epoch_idx'] == self.config.optimizer.num_epochs:
            save_fname = 'last_epoch{}_acc{:.4f}.ckpt'.format(
                val_metrics['epoch_idx'], val_metrics['acc'])
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(state_dict, save_pth)

        if val_metrics['epoch_idx'] == 'refc':
            save_fname = 'ReFc_Final_acc{:.4f}.ckpt'.format(val_metrics['acc'])
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(state_dict, save_pth)
        
        if val_metrics['epoch_idx'] == 'final_quantize':
            save_fname = 'Final_Quantize_acc{:.4f}.ckpt'.format(val_metrics['acc'])
            save_pth = os.path.join(self.output_dir, save_fname)
            torch.save(state_dict, save_pth)         

        return tag

    def summary(self):
        print('Training Completed! '
              'Best accuracy: {:.2f} '.format(100 * self.best_acc),
              flush=True)
