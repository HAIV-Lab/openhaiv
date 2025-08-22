# Benchmark: CUB200
# Model: ResNet18
# Method: Joint
# Task: Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_leanring/inc_cub_joint.yaml \
    --opts device='cuda:0'