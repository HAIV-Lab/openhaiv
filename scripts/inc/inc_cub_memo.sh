# Benchmark: CUB200
# Model: ResNet18
# Method: MEMO
# Task: Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_leanring/inc_cub_memo.yaml \
    --opts device='cuda:0'