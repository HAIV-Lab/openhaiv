# Benchmark: OES
# Model: ResNet18
# Method: LwF
# Task: Class-incremental Learning
CUDA_VISIBLE_DEVICES=4 python train.py \
    --cfg configs/pipeline/incremental_learning/inc_oes_lwf.yaml \
    --opts device='cuda:0'