from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
import argparse
import yaml


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-files", "--cfg", type=str, default='configs/detect/yolo.yaml', help="path to config file")
args = parser.parse_args()

def main():
    print(args)
    cfg = args.config_files
    with open(cfg, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    DEFAULT_CFG.save_dir = result['save_path']
    model = YOLO(result['yolo_yaml']).load(result['yolo_model'])  # build from YAML and transfer weights
    model.train(data=result['data'], epochs=result['epochs'], imgsz=result['image_size'], batch=result['batch'], workers=result['num_workers'])

if __name__ == '__main__':
    main()