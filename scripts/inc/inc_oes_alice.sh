# Benchmark: OES
# Model: ResNet18
# Method: ALICE
# Task: Few-shot Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_learning/inc_oes_alice.yaml \
    --opts device='cuda:0'