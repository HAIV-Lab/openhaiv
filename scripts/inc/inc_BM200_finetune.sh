# Benchmark: BM200
# Model: ResNet18
# Method: Finetune
# Task: Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_learning/inc_BM200_finetune.yaml \
    --opts device='cuda:0'