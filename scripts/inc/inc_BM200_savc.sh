# Benchmark: BM200
# Model: ResNet18
# Method: SAVC
# Task: Few-shot Class-incremental Learning
python ncdia/train.py \
    --cfg configs/pipeline/incremental_leanring/inc_BM200_savc.yaml \
    --opts device='cuda:0'