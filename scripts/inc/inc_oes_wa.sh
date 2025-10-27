# Benchmark: OES
# Model: ResNet18
# Method: WA
# Task: Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_learning/inc_oes_wa.yaml \
    --opts device='cuda:0'