"""
Basic Runner for Compression
version: 0.1.0
update: 2023-12-22
"""
import os
import time
import torch
from tqdm import tqdm

from utils import MovingAverageMeter, AverageMeter


class BasicRunner():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cur_epoch = 0
        self.max_epoch = 1
        self.print_freq = 5

        self._models = {}
        self.model = None
        self.optim = None
        self.sched = None

    def check_cfg(self, cfg, name: str):
        """Check configuration.
        Args:
            cfg (Configs): configuration
            name (str): name of model
        """
        if not getattr(self, 'checked', False):
            assert cfg.runner.name == name
        self.checked = True

    def register_model(self, name, model):
        """Register model.
        Args:
            name (str): name of model
            model (nn.Module): model
        """
        if '_models' not in self.__dict__:
            raise ValueError("Please call `super().__init__()` first.")

        self._models[name] = model

    def train(self, train=True):
        """Set model to training mode.
        Args:
            train (bool): training mode
        """
        for model in self._models.values():
            if train:
                model.train()
            else:
                model.eval()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.train(False)

    def parameters(self):
        """Get parameters of all registered models.
        Returns:
            generator: parameters
        """
        for model in self._models.values():
            for param in model.parameters():
                yield param

    def named_parameters(self):
        """Get named parameters of all registered models.
        Returns:
            generator: named parameters
        """
        for name, model in self._models.items():
            for n, p in model.named_parameters():
                yield f'{name}.{n}', p

    def train_step(self, batch, iters, **kwargs):
        """Train step for one batch.
        Args:
            batch (dict): batch data
            iters (int): number of iterations
        Returns:
            tuple: loss, metrics and number of samples
        """
        raise NotImplementedError

    def eval_step(self, batch, iters, **kwargs):
        """Evaluation step for one batch.
        Args:
            batch (dict): batch data
            iters (int): number of iterations
        Returns:
            tuple: loss, metrics and number of samples
        """
        raise NotImplementedError
    
    def update(self, epoch):
        """Update status of trainer for one epoch.
        Args:
            epoch (int): current epoch
        """
        if getattr(self, 'sched', None):
            self.sched.step()

    def run(self, train_loader, val_loader, **kwargs):
        """Run training stage.
        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
            val_loader (torch.utils.data.DataLoader): validation data loader
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_iters = self.max_epoch * len(train_loader)

        loss_meter = MovingAverageMeter(max_len=1000)
        acc_meter = MovingAverageMeter(max_len=1000)
        time_meter = MovingAverageMeter(max_len=1000)

        start = time.time()
        for epoch in range(self.cur_epoch, self.max_epoch):
            self.cur_epoch = epoch
            self.train()

            for i, batch in enumerate(train_loader):
                loss, acc, num = self.train_step(
                    batch, iters=i+epoch*len(train_loader), **kwargs)

                loss_meter.update(loss, num)
                acc_meter.update(acc, num)
                time_meter.update(time.time() - start)

                if (i+1) % self.print_freq == 0:
                    s = f"Epoch [{epoch+1}/{self.max_epoch}]"
                    s += f" Iter [{i+1}/{len(train_loader)}]"
                    s += f" Time [{time_meter.val:.3f} ({time_meter.avg:.3f})]"
                    s += f" Loss [{loss_meter.val:.3f} ({loss_meter.avg:.3f})]"
                    s += f" Acc [{acc_meter.val:.3f} ({acc_meter.avg:.3f})]"
                    s += f" LR [{self.optim.param_groups[0]['lr']:.2e}]" if getattr(self, 'optim', None) else ""
                    s += f" ETA [{time.strftime('%H:%M:%S', time.gmtime(time_meter.avg * ((self.max_epoch - epoch) * len(train_loader) - i)))}]" 
                    self.cfg.logger.info(s)

                start = time.time()

            self.update(epoch)
    
    def evaluate(self, data_loader, verbose=True, **kwargs):
        """Evaluate model on data loader.
        Args:
            data_loader (torch.utils.data.DataLoader): data loader
            verbose (bool): whether to print evaluation results for every batch
        Returns:
            dict: evaluation results
        """
        self.eval()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        if verbose:
            tbar = tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                dynamic_ncols=True)
        else:
            tbar = enumerate(data_loader)

        for i, batch in tbar:
            loss, acc, num = self.eval_step(batch, iters=i, **kwargs)

            loss_meter.update(loss, num)
            acc_meter.update(acc, num)

            if verbose:
                tbar.set_description(f"Evaluating: " + \
                    f"Loss [{loss_meter.avg:.3f}] " + \
                    f"Acc [{acc_meter.avg:.3f}]")

        self.cfg.logger.info(
            f"========== Evaluation ==========\n" + \
            f"Epoch [{self.cur_epoch+1}/{self.max_epoch}] " + \
            f"Loss [{loss_meter.avg:.3f}] " + \
            f"Acc [{acc_meter.avg:.3f}]" + \
            f"\n===============================")

        return {
            "loss": loss_meter.avg,
            "acc": acc_meter.avg,
        }

    def state_dict(self):
        """Get state dict of all registered models.
        Returns:
            dict: state dict
        """
        state_dict = {}
        for name, model in self._models.items():
            state_dict[name] = model.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        """Load state dict of all registered models.
        Args:
            state_dict (dict): state dict
        """
        for name, model in self._models.items():
            model.load_state_dict(state_dict[name])

    def save_checkpoint(self, checkpoint, save_dir, model_name, is_best=False):
        """Save checkpoint to file.
        Args:
            checkpoint (dict): checkpoint
            save_dir (str): directory to save checkpoint
            model_name (str): checkpoint name
        """
        os.makedirs(save_dir, exist_ok=True)

        fpath = os.path.join(save_dir, model_name)
        torch.save(checkpoint, fpath)
        self.cfg.logger.info(f"Saving checkpoint to {fpath}")

        if is_best:
            fpath = os.path.join(save_dir, "ckpt_best.pth")
            torch.save(checkpoint, fpath)
            self.cfg.logger.info(f"Saving current best to {fpath}")
            self.cfg.runner.best = os.path.abspath(os.path.expanduser(fpath))

    def load_checkpoint(self, fpath, device='cpu'):
        """Load checkpoint from file.
        Args:
            fpath (str): checkpoint
            device (str): device to load checkpoint
        """
        if not os.path.exists(fpath):
            raise ValueError(f"Checkpoint {fpath} not found!")
        
        checkpoint = torch.load(fpath, map_location=device)
        self.load_state_dict(checkpoint['state_dict'])
        if self.optim and checkpoint['optimizer']:
            self.optim.load_state_dict(checkpoint['optimizer'])
        if self.sched and checkpoint['scheduler']:
            self.sched.load_state_dict(checkpoint['scheduler'])
        self.cur_epoch = checkpoint['epoch']

        self.cfg.logger.info(
            f"Checkpoint loaded from {fpath}\n" + \
            f"Current epoch: {self.cur_epoch}\n" + \
            f"Evaluated result: {checkpoint['eval_result']}")

    def save_model(self, eval_result=None, epoch=None, save_dir=''):
        """Save model to file.
        Args:
            eval_result (dict): evaluation result
            epoch (int): current epoch
            save_dir (str): directory to save model
        """
        if epoch is None:
            epoch = self.cur_epoch + 1
        
        if not save_dir:
            save_dir = os.path.join(self.cfg.output_dir, "checkpoints")

        is_best=False
        if eval_result is not None:
            if 'best_result' not in self.__dict__ or \
                eval_result['acc'] > self.best_result['acc']:
                self.best_result = eval_result
                is_best = True

        self.save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': self.state_dict(),
                'optimizer': self.optim.state_dict() if self.optim else None,
                'scheduler': self.sched.state_dict() if self.sched else None,
                'eval_result': eval_result,
            },
            save_dir=save_dir,
            model_name=f"ckpt_epoch{epoch}.pth",
            is_best=is_best
        )

    def load_model(self, fpath, device='cpu'):
        """Load model from file.
        Args:
            fpath (str): checkpoint
            device (str): device to load checkpoint
        """
        self.load_checkpoint(fpath, device=device)

    def __call__(self, train_loader, val_loader):
        return self.run(train_loader, val_loader)
