# Benchmark: BM200
# Model: ResNet18
# Method: ALICE
# Task: Few-shot Class-incremental Learning
python ncdia/train.py \
    --cfg configs/pipeline/incremental_leanring/inc_BM200_alice.yaml \
    --opts device='cuda:0'