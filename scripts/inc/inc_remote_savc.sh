# Benchmark: Remote
# Model: ResNet18
# Method: SAVC
# Task: Few-shot Class-incremental Learning
python ncdia/train.py \
    --cfg configs/pipeline/incremental_leanring/inc_remote_savc.yaml \
    --opts device='cuda:0'
