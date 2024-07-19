import argparse
from utils.cfg import setup_cfg
from utils.tools import set_random_seed
from trainers import PreTrainer, IncTrainer
from torchvision.models import resnet18


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
    # trainer = PreTrainer(cfg) if cfg.session == 0 else IncTrainer(cfg)
    model = resnet18(pretrained=True)
    trainer = PreTrainer(model, cfg)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main(args)
