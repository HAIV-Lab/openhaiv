# Benchmark: CUB200
# Model: ResNet18
# Method: Coil
# Task: Class-incremental Learning
python ncdia/train.py \
    --cfg configs/pipeline/incremental_leanring/inc_cub_coil.yaml \
    --opts device='cuda:0'