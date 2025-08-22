# Benchmark: CUB200
# Model: ResNet18
# Method: LwF
# Task: Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_leanring/inc_cub_lwf.yaml \
    --opts device='cuda:0'