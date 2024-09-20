import argparse
from ncdia.utils.cfg import setup_cfg
from ncdia.utils.tools import set_random_seed
from ncdia.utils import TRAINERS


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-files", "--cfg", type=str, nargs="+", default=[], help="path to config file")
parser.add_argument(
    "--opts",
    default=[],
    nargs=argparse.REMAINDER,
    help="modify config options using the command line"
)
args = parser.parse_args()


def main(args):
    # Setup config
    cfg = setup_cfg(args)

    # Setup random seed
    if cfg.seed >= 0:
        set_random_seed(cfg.seed)

    # Build the trainer from config
    trainer = TRAINERS.build(cfg.trainer, cfg=cfg)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main(args)
