# Benchmark: ImageNet-R
# Model: ResNet18
# Method: Alice
# Task: Few-shot Class-incremental Learning
python train.py \
    --cfg configs/pipeline/incremental_leanring/inc_imagenetr_alice.yaml \
    --opts device='cuda:0'
