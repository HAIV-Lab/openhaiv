# Benchmark: Remote
# Model: ResNet18
# Method: Alice
# Task: Few-shot Class-incremental Learning
python ncdia/train.py \
    --cfg configs/pipeline/incremental_leanring/inc_remote_alice.yaml \
    --opts device='cuda:0'