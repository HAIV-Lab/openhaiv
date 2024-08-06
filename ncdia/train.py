import argparse
from ncdia.utils.cfg import setup_cfg
from ncdia.utils.tools import set_random_seed
from ncdia.utils import TRAINERS
# from ncdia.trainers import PreTrainer, IncTrainer
# from ncdia.algorithms.ncd import AutoNCD


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
    trainer = TRAINERS.build(dict(cfg.trainer), cfg=cfg)

    # Start training
    trainer.train()

    # _, pre_train_loader, pre_test_loader = cli_dataloader(cfg, session-1)
    # _, train_loader, test_loader = cli_dataloader(cfg, session)

    # ncd_detecter = AutoNCD(
    #     trainer.model,
    #     pre_train_loader, pre_test_loader,
    #     cfg.CIL.base_classes, cfg.CIL.way,
    #     session, trainer.device, verbose=True,
    # )
    # train_loader = ncd_detecter.relabel(train_loader, metrics=['msp'], prec_th=0.42)


if __name__ == '__main__':
    main(args)
