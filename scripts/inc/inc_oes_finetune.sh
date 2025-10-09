# Benchmark: OES
# Model: ResNet18
# Method: Finetune
# Task: Class-incremental Learning
CUDA_VISIBLE_DEVICES=2 python train.py \
    --cfg configs/pipeline/incremental_learning/inc_oes_finetune.yaml \
    --opts device='cuda:0'