# Benchmark: BMF
# Model: ResNet18
# Method: GEM
# Task: Class-incremental Learning
python ncdia/train.py \
    --cfg configs/pipeline/incremental_leanring/inc_bmf_gem.yaml \
    --opts device='cuda:0'