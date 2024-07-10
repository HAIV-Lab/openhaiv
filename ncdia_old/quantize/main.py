import argparse
import torch

from utils import *
from runner import execute_runner


def setup_cfg(args):
    cfg = Configs()

    # Set default vaules
    cfg.merge_from_list([
        'use_cuda=True',
        'gpus=0',
        'seed=-1',
        'output_dir="./results/"',
    ])

    # Load config from file
    for opt in args.opts:
        if opt.startswith('_base_'):
            cfg.merge_from_yaml(opt.split('=')[1])
            args.opts.remove(opt)

    for cfg_file in args.config_files:
        cfg.merge_from_yaml(cfg_file)

    # Load config from input arguments
    cfg.merge_from_dict(vars(args))

    # Load config from command line
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def setup_logger(cfg):
    logger = Logger(cfg.output_dir)

    # Create config file and save to disk in the form of yaml
    logger.create_config(cfg['cfg'])

    # Print config and save to log file
    logger.info(cfg)

    cfg.logger = logger

    return logger


def main(args):
    # Setup config
    cfg = setup_cfg(args)

    # Setup logger
    setup_logger(cfg)

    # Setup random seed
    if cfg.seed >= 0:
        set_random_seed(cfg.seed)
        cfg.logger.info(f"Setting fixed seed: {cfg.seed}")

    # Setup device
    if torch.cuda.is_available() and cfg.use_cuda:
        torch.backends.cudnn.benchmark = True
        cfg.device = torch.device(f"cuda:{cfg.gpus}")
    else:
        cfg.device = torch.device("cpu")

    # Execute trainer
    execute_runner(cfg, args.eval_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-files", "--cfg", type=str, nargs="+", default=[], help="path to config file")
    parser.add_argument(
        "--eval-only", "--eval", action="store_true", default=False, help="perform evaluation only")
    parser.add_argument(
        "--opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="modify config options using the command line"
    )
    args = parser.parse_args()
    main(args)
