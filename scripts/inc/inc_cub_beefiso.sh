# Benchmark: CUB200
# Model: ResNet18
# Method: BeefIso
# Task: Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_leanring/inc_cub_beefiso.yaml \
    --opts device='cuda:0'