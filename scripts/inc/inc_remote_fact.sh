# Benchmark: Remote
# Model: ResNet18
# Method: FACT
# Task: Few-shot Class-incremental Learning
python ncdia/train.py \
    --cfg configs/pipeline/incremental_leanring/inc_remote_fact.yaml \
    --opts device='cuda:0'
