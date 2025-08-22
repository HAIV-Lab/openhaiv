# Benchmark: BM200
# Model: ResNet18
# Method: BiC
# Task: Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_leanring/inc_BM200_bic.yaml \
    --opts device='cuda:0'