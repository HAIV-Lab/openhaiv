import os
from tqdm import tqdm
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler

from ncdia.utils import (
    TRAINERS, HOOKS, LOSSES, ALGORITHMS,
    mkdir_if_missing, auto_device,
)
from .hooks import Hook
from .priority import get_priority, Priority
from .optims import build_optimizer, build_scheduler
from ncdia.datasets import build_dataloader


@TRAINERS.register
class BaseTrainer(object):
    """Basic trainer class for training models.

    Args:
        model (nn.Module): Model to be trained.
        cfg (dict, optional): Configuration for trainer, Contains:
            - 'algorithm' (dict):
                - 'type' (str): Type of algorithm.
            - `max_epochs` (int): Total epochs for training.
            - 'criterion' (dict):
                - 'type' (str): Type of criterion for training.
            - 'optimizer':
                - 'type' (str): Name of optimizer.
                - 'param_groups' (dict | None): If provided, directly optimize
                    param_groups and abandon model.
                - kwargs (dict) for optimizer, such as 'lr', 'weight_decay', etc.
            - 'scheduler':
                - 'type' (str): Name of scheduler.
                - kwargs (dict) for scheduler, such as 'step_size', 'gamma', etc.
            - 'device' (str | torch.device | None): Device to use.
                If None, use 'cuda' if available.
            - 'trainloader':
                - 'dataset': 
                    - 'type' (str): Type of dataset.
                    - kwargs (dict) for dataset, such as 'root', 'split', etc.
                - kwargs (dict) for DataLoader, such as 'batch_size', 'shuffle', etc.
            - 'valloader':
                - 'dataset': 
                    - 'type' (str): Type of dataset.
                    - kwargs (dict) for dataset, such as 'root', 'split', etc.
                - kwargs (dict) for DataLoader, such as 'batch_size', 'shuffle', etc.
            - 'testloader':
                - 'dataset':
                    - 'type' (str): Type of dataset.
                    - kwargs (dict) for dataset, such as 'root', 'split', etc.
                - kwargs (dict) for DataLoader, such as 'batch_size', 'shuffle', etc.
            - 'exp_name' (str): Experiment name.
            - 'work_dir' (str): Working directory to save logs and checkpoints.
        train_loader (DataLoader | dict, optional): DataLoader for training.
        val_loader (DataLoader | dict, optional): DataLoader for validation.
        test_loader (DataLoader | dict, optional): DataLoader for testing.
        default_hooks (dict, optional): Default hooks to be registered.
        custom_hooks (list, optional): Custom hooks to be registered.
        load_from (str, optional): Checkpoint file path to load.
        work_dir (str, optional): Working directory to save logs and checkpoints.

    Attributes:
        model (nn.Module): Neural network models.
        train_loader (DataLoader): DataLoader for training.
        val_loader (DataLoader): DataLoader for validation.
        test_loader (DataLoader): DataLoader for testing.
        optimizer (Optimizer): Optimizer.
        scheduler (lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion (Callable): Criterion for training.
        algorithm (object): Algorithm for training.
        max_epochs (int): Total epochs for training.
        cfg (dict): Configuration for trainer.
        hooks (List[Hook]): List of registered hooks.
        logger (Logger): Logger for logging information.
        device (torch.device): Device to use.
        work_dir (str): Working directory to save logs and checkpoints.
        exp_name (str): Experiment name.
        load_from (str): Checkpoint file path to load.

    """
    def __init__(
            self,
            model: nn.Module,
            cfg: dict | None = None,
            train_loader: DataLoader | dict | None = None,
            val_loader: DataLoader | dict | None = None,
            test_loader: DataLoader | dict | None = None,
            default_hooks: Dict[str, Hook | dict] | None = None,
            custom_hooks: List[Hook | dict] | None = None,
            load_from: str | None = None,
            exp_name: str | None = None,
            work_dir: str | None = None,
    ):
        super(BaseTrainer, self).__init__()
        self._model = model
        self._cfg = cfg

        if 'optimizer' in self._cfg:
            self._optimizer = dict(self._cfg['optimizer'])
        else:
            raise KeyError('Optimizer is not found in `cfg`.')
        
        if 'scheduler' in self._cfg:
            self._scheduler = dict(self._cfg['scheduler'])
        else:
            self._scheduler = {'name': 'constant'}
        
        if 'criterion' in self._cfg:
            self._criterion = dict(self._cfg['criterion'])
        else:
            raise KeyError("Criterion is not found in `cfg`.")
        
        if 'algorithm' in self._cfg:
            self._algorithm = dict(self._cfg['algorithm'])
        else:
            raise KeyError("Algorithm is not found in `cfg`.")
        
        self._train_loader = {}
        if 'trainloader' in self._cfg:
            self._train_loader.update(dict(self._cfg['trainloader']))
        if isinstance(train_loader, dict):
            self._train_loader.update(train_loader)
        elif isinstance(train_loader, DataLoader):
            self._train_loader = train_loader
        if not self._train_loader:
            raise KeyError("Trainloader is not found in `cfg`.")
        
        self._val_loader = {}
        if 'valloader' in self._cfg:
            self._val_loader.update(dict(self._cfg['valloader']))
        if isinstance(val_loader, dict):
            self._val_loader.update(val_loader)
        elif isinstance(val_loader, DataLoader):
            self._val_loader = val_loader
        if not self._val_loader:
            raise KeyError("Valloader is not found in `cfg`.")
        
        self._test_loader = {}
        if 'testloader' in self._cfg:
            self._test_loader.update(dict(self._cfg['testloader']))
        if isinstance(test_loader, dict):
            self._test_loader.update(test_loader)
        elif isinstance(test_loader, DataLoader):
            self._test_loader = test_loader
        if not self._test_loader:
            raise KeyError("Testloader is not found in `cfg`.")

        # load checkpoint
        self.load_from = load_from

        # work directory
        self._work_dir = work_dir or str(self._cfg['work_dir']) or 'output'
        self._exp_name = exp_name or str(self._cfg['exp_name']) or 'exp'
        mkdir_if_missing(self.work_dir)

        self._hooks: List[Hook] = []
        # register hooks to `self._hooks`
        self.register_hooks(default_hooks, custom_hooks)
        # call hooks to initialize trainer
        self.call_hook('init_trainer')
        # log hooks information
        self.logger.write(f'Hooks will be executed in the following '
                          f'order:\n{self.get_hooks_info()}')
    
    @property
    def cfg(self) -> object:
        """Configs: Configuration for trainer."""
        return self._cfg

    @property
    def hooks(self) -> List[Hook]:
        """List[Hook]: List of registered hooks."""
        return self._hooks
    
    @property
    def logger(self) -> object:
        """Logger: Logger for logging information."""
        if '_logger' not in self.__dict__:
            return None
        else:
            return self._logger
    
    @property
    def work_dir(self) -> str:
        """str: Working directory to save logs and checkpoints."""
        return os.path.join(self._work_dir, self._exp_name)        
    
    @property
    def model(self) -> nn.Module:
        """nn.Module: Model to be trained."""
        return self._model
    
    @property
    def train_loader(self) -> DataLoader:
        """DataLoader: DataLoader for training."""
        if isinstance(self._train_loader, dict):
            self._train_loader = build_dataloader(self._train_loader)
        return self._train_loader
    
    @property
    def val_loader(self) -> DataLoader:
        """DataLoader: DataLoader for validation."""
        if isinstance(self._val_loader, dict):
            self._val_loader = build_dataloader(self._val_loader)
        return self._val_loader
    
    @property
    def test_loader(self) -> DataLoader:
        """DataLoader: DataLoader for testing."""
        if isinstance(self._test_loader, dict):
            self._test_loader = build_dataloader(self._test_loader)
        return self._test_loader
    
    @property
    def optimizer(self) -> Optimizer:
        """Optimizer: Optimizer to optimize model parameters."""
        if isinstance(self._optimizer, dict):
            self._optimizer = build_optimizer(self._optimizer, self.model)
        return self._optimizer
    
    @property
    def scheduler(self) -> lr_scheduler._LRScheduler:
        """lr_scheduler._LRScheduler: Learning rate scheduler."""
        if isinstance(self._scheduler, dict):
            self._scheduler = build_scheduler(self._scheduler, self.optimizer)
        return self._scheduler
    
    @property
    def criterion(self) -> nn.Module:
        """Callable: Criterion for training."""
        if isinstance(self._criterion, dict):
            self._criterion = LOSSES.build(self._criterion)
        return self._criterion
    
    @property
    def algorithm(self) -> object:
        """object: Algorithm for training."""
        if isinstance(self._algorithm, dict):
            self._algorithm = ALGORITHMS.build(self._algorithm)
        return self._algorithm
    
    @property
    def max_epochs(self) -> int:
        """int: Total epochs for training."""
        if not hasattr(self, '_max_epochs'):
            if not 'max_epochs' in self._cfg:
                self._max_epochs = 1
            else:
                self._max_epochs = int(self._cfg['max_epochs'])

        return self._max_epochs
    
    @property
    def device(self) -> torch.device:
        """torch.device: Device to use."""
        if not hasattr(self, '_device'):
            if not 'device' in self._cfg:
                _device = None
            else:
                _device = self._cfg['device']
            self._device = auto_device(_device)
        
        return self._device
    
    def train_step(self, batch, **kwargs):
        """Training step. This method should be implemented in subclasses.
        
        Args:
            batch (dict | tuple | list): A batch of data from the data loader.

        Returns:
            results (dict): Contains the following:
                {
                    "key1": value1,
                    "key2": value2,
                    ...
                }
                keys denote the description of the value, such as "loss", "acc", "ccr", etc.
                values are the corresponding values of the keys, can be int, float, str, etc.
        """
        raise NotImplementedError
    
    def val_step(self, batch, **kwargs):
        """Validation step. This method should be implemented in subclasses.
        
        Args:
            batch (dict | tuple | list): A batch of data from the data loader.

        Returns:
            results (dict): Contains the following:
                {
                    "key1": value1,
                    "key2": value2,
                    ...
                }
                keys denote the description of the value, such as "loss", "acc", "ccr", etc.
                values are the corresponding values of the keys, can be int, float, str, etc.
        """
        raise NotImplementedError
    
    def test_step(self, batch, **kwargs):
        """Test step. This method should be implemented in subclasses.
        
        Args:
            batch (dict | tuple | list): A batch of data from the data loader.

        Returns:
            results (dict): Contains the following:
                {
                    "key1": value1,
                    "key2": value2,
                    ...
                }
                keys denote the description of the value, such as "loss", "acc", "ccr", etc.
                values are the corresponding values of the keys, can be int, float, str, etc.
        """
        raise NotImplementedError

    def train(self) -> nn.Module:
        """Launch the training process.

        Returns:
            model (nn.Module): Trained model.
        """
        self.call_hook('before_run')

        if self.load_from:
            self.load_ckpt(self.load_from)
            self.call_hook('after_load_checkpoint')

        self.call_hook('before_train')

        for epoch in range(self.max_epochs):
            self.call_hook('before_train_epoch')

            tbar = tqdm(
                self.train_loader,
                desc=f'Epoch {epoch+1}/{self.max_epochs}',
                dynamic_ncols=True)

            for batch in tbar:
                self.call_hook('before_train_iter')

                self.train_step(batch)

                self.call_hook('after_train_iter')

            self.call_hook('after_train_epoch')

            self.val()

        self.call_hook('before_save_checkpoint')
        self.save_ckpt(os.path.join(self.work_dir, 'latest.pth'))

        self.call_hook('after_train')

        self.test()

        self.call_hook('after_run')

        return self.model

    def val(self):
        """Validation process."""
        self.call_hook('before_val')
        self.call_hook('before_val_epoch')

        tbar = tqdm(self.val_loader, desc='Validation', dynamic_ncols=True)
        for batch in tbar:
            self.call_hook('before_val_iter')
            self.val_step(batch)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')
        self.call_hook('after_val')

    def test(self):
        """Test process."""
        self.call_hook('before_test')
        self.call_hook('before_test_epoch')

        tbar = tqdm(self.test_loader, desc='Testing', dynamic_ncols=True)
        for batch in tbar:
            self.call_hook('before_test_iter')
            self.test_step(batch)
            self.call_hook('after_test_iter')

        self.call_hook('after_test_epoch')
        self.call_hook('after_test')
    
    def load_ckpt(self, fpath: str, device: str| None = 'cpu'):
        """Load checkpoint from file.

        Args:
            fpath (str): Checkpoint file path.
            device (str, optional): Device to load checkpoint. Defaults to 'cpu'.

        Returns:
            model (nn.Module): Loaded model.

        Raises:
            ValueError: Checkpoint not found when `fpath` does not exist.
        """
        if not os.path.exists(fpath):
            raise ValueError(f'Checkpoint {fpath} not found!')
        
        checkpoint = torch.load(fpath, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.logger.info(f'Checkpoint loaded from {fpath}')

        return self.model
    
    def save_ckpt(self, fpath: str):
        """Save checkpoint to file.
        
        Args:
            fpath (str): Checkpoint file path.
        """
        mkdir_if_missing(os.path.dirname(fpath))

        checkpoint = {
            'state_dict': self.model.state_dict()}
        torch.save(checkpoint, fpath)

        self.logger.info(f'Checkpoint saved to {fpath}')

    def call_hook(
            self,
            fn_name: str,
            **kwargs) -> None:
        """Call all hooks with the specified function name.

        Args:
            fn_name (str): Function name to be called, such as
                'before_train_epoch', 'after_train_epoch', 
                'before_train_iter', 'after_train_iter', 
                'before_val_epoch', 'after_val_epoch', 
                'before_val_iter', 'after_val_iter'.
            kwargs (dict): Arguments for the function.
        """
        for hook in self._hooks:
            # support adding additional custom hook methods
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(self, **kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from None
    
    def register_hook(
            self,
            hook: Hook | dict,
            priority: str | int | None = None) -> None:
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Priority of hook will be decided with the following priority:

        - ``priority`` argument. If ``priority`` is given, it will be priority
          of hook.
        - If ``hook`` argument is a dict and ``priority`` in it, the priority
          will be the value of ``hook['priority']``.
        - If ``hook`` argument is a dict but ``priority`` not in it or ``hook``
          is an instance of ``hook``, the priority will be ``hook.priority``.

        Args:
            hook (:obj:`Hook` or dict): The hook to be registered.
            priority (int or str or :obj:`Priority`, optional): Hook priority.
                Lower value means higher priority.
        """
        if not isinstance(hook, (Hook, dict)):
            raise TypeError(
                f'hook should be an instance of Hook or dict, but got {hook}')

        _priority = None
        if isinstance(hook, dict):
            if 'priority' in hook:
                _priority = hook.pop('priority')

            hook_obj = HOOKS.build(hook)
        else:
            hook_obj = hook

        if priority is not None:
            hook_obj.priority = priority
        elif _priority is not None:
            hook_obj.priority = _priority

        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if get_priority(hook_obj.priority) >= get_priority(
                    self._hooks[i].priority):
                self._hooks.insert(i + 1, hook_obj)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook_obj)

    def register_default_hooks(
            self,
            hooks: Dict[str, Hook | dict] | None = None) -> None:
        """Register default hooks into hook list.

        ``hooks`` will be registered into runner to execute some default
        actions like updating model parameters or saving checkpoints.

        Default hooks and their priorities:

        ```
        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | RuntimeInfoHook      | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | IterTimerHook        | NORMAL (50)             |
        +----------------------+-------------------------+
        | DistSamplerSeedHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | LoggerHook           | BELOW_NORMAL (60)       |
        +----------------------+-------------------------+
        | ParamSchedulerHook   | LOW (70)                |
        +----------------------+-------------------------+
        | CheckpointHook       | VERY_LOW (90)           |
        +----------------------+-------------------------+
        ```

        If ``hooks`` is None, above hooks will be registered by
        default::

            default_hooks = dict(
                runtime_info=dict(type='RuntimeInfoHook'),
                timer=dict(type='IterTimerHook'),
                sampler_seed=dict(type='DistSamplerSeedHook'),
                logger=dict(type='LoggerHook'),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(type='CheckpointHook', interval=1),
            )

        If not None, ``hooks`` will be merged into ``default_hooks``.
        If there are None value in default_hooks, the corresponding item will
        be popped from ``default_hooks``::

            hooks = dict(timer=None)

        The final registered default hooks will be :obj:`RuntimeInfoHook`,
        :obj:`DistSamplerSeedHook`, :obj:`LoggerHook`,
        :obj:`ParamSchedulerHook` and :obj:`CheckpointHook`.

        Args:
            hooks (dict[str, Hook or dict], optional): Default hooks or configs
                to be registered.
        """
        default_hooks: dict = dict(
            # runtime_info=dict(type='RuntimeInfoHook'),
            # timer=dict(type='IterTimerHook'),
            # sampler_seed=dict(type='DistSamplerSeedHook'),
            # param_scheduler=dict(type='ParamSchedulerHook'),
            # checkpoint=dict(type='CheckpointHook', interval=1),

            logger=dict(type='LoggerHook'),
            model=dict(type='ModelHook'),
            optimizer = dict(type='OptimizerHook'),
            scheduler = dict(type='SchedulerHook'),
        )
        if hooks is not None:
            for name, hook in hooks.items():
                if name in default_hooks and hook is None:
                    # remove hook from _default_hooks
                    default_hooks.pop(name)
                else:
                    assert hook is not None
                    default_hooks[name] = hook

        for hook in default_hooks.values():
            self.register_hook(hook)

    def register_custom_hooks(self, hooks: List[Hook | dict]) -> None:
        """Register custom hooks into hook list.

        Args:
            hooks (list[Hook | dict]): List of hooks or configs to be
                registered.
        """
        for hook in hooks:
            self.register_hook(hook)

    def register_hooks(
            self,
            default_hooks: Dict[str, Hook | dict] | None = None,
            custom_hooks: List[Hook | dict] | None = None) -> None:
        """Register default hooks and custom hooks into hook list.

        Args:
            default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks
                to execute default actions like updating model parameters and
                saving checkpoints.  Defaults to None.
            custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
                custom actions like visualizing images processed by pipeline.
                Defaults to None.
        """
        self.register_default_hooks(default_hooks)

        if custom_hooks is not None:
            self.register_custom_hooks(custom_hooks)

    def get_hooks_info(self) -> str:
        """Get registered hooks information.

        Returns:
            info (str): Information of registered hooks.
        """
        # Get hooks info in each stage
        stage_hook_map: Dict[str, list] = {stage: [] for stage in Hook.stages}
        for hook in self.hooks:
            try:
                priority = Priority(hook.priority).name  # type: ignore
            except ValueError:
                priority = hook.priority  # type: ignore
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)
