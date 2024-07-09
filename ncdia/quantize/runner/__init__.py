"""
Runner
version: 0.1.1
update: 2024-01-15
"""
from utils import Configs, Register
from dataset import build_dataloader

from .ptq import PTQ
from .adaround import AdaRound
from .qat import QAT

RUNNERS = Register({
    'ptq': PTQ,
    'adaround': AdaRound,
    'qat': QAT,
})


def build_runner(cfg: Configs, checkpoint=None):
    """ Build runner from configs.
    Args:
        cfg (Configs): global Configs object
        checkpoint (str): path to checkpoint
    Returns:
        Runner: Runner.
    """
    try:
        runner = RUNNERS[cfg.runner.name](cfg)
    except KeyError:
        raise NotImplementedError(f"Runner {cfg.runner.name} is not implemented.")
    
    if checkpoint:
        runner.load_checkpoint(checkpoint, device='cpu')
    
    return runner


def execute_runner(cfg: Configs, eval_only=False):
    """ Build runner and execute training and testing."""

    if not eval_only:

        # build train and val dataloader
        train_loader = build_dataloader(cfg.train_dataset, cfg.train_loader)
        val_loader = build_dataloader(cfg.val_dataset, cfg.val_loader)

        # build runner
        cfg.model.classnames = train_loader.dataset.classnames
        cfg.model.num_classes = train_loader.dataset.num_classes
        runner = build_runner(cfg, cfg.runner.checkpoint)

        # execute training
        runner(train_loader, val_loader)
        cfg.runner.verbose = False


    if cfg.test_dataset:

        # build test dataloader
        test_loader = build_dataloader(cfg.test_dataset, cfg.test_loader)

        # build runner
        cfg.model.classnames = test_loader.dataset.classnames
        cfg.model.num_classes = test_loader.dataset.num_classes
        runner = build_runner(cfg, cfg.runner.best or cfg.runner.checkpoint)

        # execute testing
        runner.evaluate(test_loader, quantized=cfg.runner.best or cfg.runner.checkpoint)

    else:
        cfg.logger.info("No test dataset is provided. Skip testing.")
