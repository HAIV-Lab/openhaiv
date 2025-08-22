# Benchmark: CUB200
# Model: ResNet18
# Method: FACT
# Task: Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_leanring/inc_cub_fact.yaml \
    --opts device='cuda:0'