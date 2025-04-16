from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
import argparse
import yaml
from ncdia.utils.cfg import setup_cfg
from ncdia.utils.tools import set_random_seed
from ncdia.trainers import PreTrainer, IncTrainer
# from torchvision.models import resnet18
# from ncdia.algorithms.ood.autoood import AutoOOD
# from ncdia.algorithms.ncd.ncd_discover import NCDDiscover
from ncdia.datasets.utils import get_dataloader
from ncdia.inference.datasets import generate_classify_datasets


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-files", "--cfg", type=str, nargs="+", default=['configs/supervised/sl_remote_alice.yaml', 'configs/inference/remote.yaml'], help="path to config file")
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

    if cfg.seed >= 0:
        set_random_seed(cfg.seed)
    with open(cfg.config_files[1], 'r', encoding='utf-8') as f:
        cfg_det = yaml.load(f.read(), Loader=yaml.FullLoader)
    model_det = YOLO(cfg_det['model_detect'])
    cli_dataloader = get_dataloader(config=cfg)
    cfg.max_epochs = 0
    num_session = cfg.num_session or 1
    for session in range(num_session):
        if session == 0:
            img_list, gt_list, pred_num = generate_classify_datasets(model_det, cfg, cfg.det_data_0, cfg.xml_data_0)
            _, train_loader, test_loader = cli_dataloader(cfg, 0)
            max_cls = max(test_loader.dataset.targets)
            indices_to_remove = [index for index, num in enumerate(gt_list) if num > max_cls]
            gt_list = [num for num in gt_list if num <= max_cls]
            img_list = [elem for index, elem in enumerate(img_list) if index not in indices_to_remove]
            test_loader.dataset.data = img_list
            test_loader.dataset.targets = gt_list
            trainer = PreTrainer(
                None, cfg,
                session=0,
                train_loader=train_loader,
                val_loader=test_loader,
                test_loader=test_loader,
            )
            trainer.load_ckpt(cfg.ckpt_0)
            # Start training
            trainer.train()
        else:
            img_list, gt_list, pred_num = generate_classify_datasets(model_det, cfg, cfg.det_data_1, cfg.xml_data_1)
            _, train_loader, test_loader = cli_dataloader(cfg, session)
            max_cls = max(test_loader.dataset.targets)
            indices_to_remove = [index for index, num in enumerate(gt_list) if num > max_cls]
            gt_list = [num for num in gt_list if num <= max_cls]
            img_list = [elem for index, elem in enumerate(img_list) if index not in indices_to_remove]
            test_loader.dataset.data = img_list
            test_loader.dataset.targets = gt_list
            trainer = IncTrainer(
                trainer.model, cfg,
                session=session,
                train_loader=train_loader,
                val_loader=test_loader,
                test_loader=test_loader,
            )
            trainer.load_ckpt(cfg.ckpt_1)
            # Start training
            trainer.train()

if __name__ == '__main__':
    main(args)