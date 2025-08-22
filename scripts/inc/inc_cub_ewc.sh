# Benchmark: CUB200
# Model: ResNet18
# Method: EWC
# Task: Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_leanring/inc_cub_ewc.yaml \
    --opts device='cuda:0'