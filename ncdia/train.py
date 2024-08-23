import argparse
from utils.cfg import setup_cfg
from utils.tools import set_random_seed
from trainers import PreTrainer, IncTrainer
from torchvision.models import resnet18
# from ncdia.algorithms.incremental.net.savc_net import SAVCNET
# from ncdia.algorithms.incremental.net.fact_net import FACTNET
# from ncdia.algorithms.incremental.net.alice_net import AliceNET
from ncdia.algorithms.ood.autoood import AutoOOD
from ncdia.algorithms.ncd.ncd_discover import NCDDiscover
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
            ood_detecter = AutoOOD(cfg.device, cfg)
            ncd_detecter = NCDDiscover(cfg)
            _, pre_train_loader, pre_test_loader = cli_dataloader(cfg, session-1)
            _, train_loader, test_loader = cli_dataloader(cfg, session)
            ood_detecter.eval(trainer.model, pre_train_loader, pre_test_loader, train_loader, session)
            ncd_dataloader = ncd_detecter.get_pseudo_newloader(trainer.model, {'train': pre_train_loader, 'test': pre_test_loader}, train_loader, pre_train_loader.dataset.transform, session-1)
            # train_loader = ncd_dataloader

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
