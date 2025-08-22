# Benchmark: BMF
# Model: ResNet18
# Method: Foster
# Task: Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_leanring/inc_bmf_foster.yaml \
    --opts device='cuda:0'