# Benchmark: BM200
# Model: ResNet18
# Method: FACT
# Task: Few-shot Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_leanring/inc_BM200_fact.yaml \
    --opts device='cuda:0'