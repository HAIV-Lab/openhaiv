# Benchmark: CUB200
# Model: ResNet18
# Method: Alice
# Task: Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_leanring/inc_cub_alice.yaml \
    --opts device='cuda:0'