# Benchmark: BMF
# Model: ResNet18
# Method: PASS
# Task: Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_leanring/inc_bmf_pass.yaml \
    --opts device='cuda:0'