# Benchmark: OES
# Model: ResNet18
# Method: EWC
# Task: Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_learning/inc_oes_ewc.yaml \
    --opts device='cuda:0'