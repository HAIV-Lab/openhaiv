# Benchmark: OES
# Model: ResNet18
# Method: Joint
# Task: Class-incremental Learning
CUDA_VISIBLE_DEVICES=3 python train.py \
    --cfg configs/pipeline/incremental_learning/inc_oes_joint.yaml \
    --opts device='cuda:0'