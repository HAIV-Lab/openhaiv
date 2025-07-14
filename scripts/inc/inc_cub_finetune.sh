# Benchmark: CUB200
# Model: ResNet18
# Method: Finetune
# Task: Class-incremental Learning
python ncdia/train.py \
    --cfg configs/pipeline/incremental_leanring/inc_cub_finetune.yaml \
    --opts device='cuda:0'