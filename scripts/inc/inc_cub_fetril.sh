# Benchmark: CUB200
# Model: ResNet18
# Method: FeTrIL
# Task: Class-incremental Learning
python ncdia/train.py \
    --cfg configs/pipeline/incremental_leanring/inc_cub_fetril.yaml \
    --opts device='cuda:0'