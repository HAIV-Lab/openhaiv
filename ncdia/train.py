import argparse
from utils.cfg import setup_cfg
from utils.tools import set_random_seed
from trainers import PreTrainer, IncTrainer
from ncdia.algorithms.ncd import AutoNCD
from ncdia.datasets.utils import get_dataloader


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
    cli_dataloader = get_dataloader(config=cfg)

    num_session = cfg.num_session or 1
    for session in range(num_session):
        if session == 0:
            _, train_loader, test_loader = cli_dataloader(cfg, 0)
            trainer = PreTrainer(
                None, cfg,
                session=0,
                train_loader=train_loader,
                val_loader=test_loader,
                test_loader=test_loader,
            )
            # Start training
            trainer.train()
        else:
            cfg.max_epochs = cfg.inc_epochs
            _, pre_train_loader, pre_test_loader = cli_dataloader(cfg, session-1)
            _, train_loader, test_loader = cli_dataloader(cfg, session)

            ncd_detecter = AutoNCD(
                trainer.model,
                pre_train_loader, pre_test_loader,
                cfg.CIL.base_classes, cfg.CIL.way,
                session, trainer.device, verbose=True,
            )
            train_loader = ncd_detecter.relabel(train_loader, metrics=['msp'], prec_th=0.42)

            trainer = IncTrainer(
                trainer.model, cfg,
                session=session,
                train_loader=train_loader,
                val_loader=test_loader,
                test_loader=test_loader,
            )
            # Start training
            trainer.train()


if __name__ == '__main__':
    main(args)
