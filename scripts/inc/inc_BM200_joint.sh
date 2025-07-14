# Benchmark: BM200
# Model: ResNet18
# Method: Joint
# Task: Class-incremental Learning
python ncdia/train.py \
    --cfg configs/pipeline/incremental_leanring/inc_BM200_Joint.yaml \
    --opts device='cuda:0'