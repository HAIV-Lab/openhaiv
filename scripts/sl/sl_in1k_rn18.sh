# Benchmark: Imagenet-1k
# Model: ResNet18
# Method: Cross-Entropy
# Task: Supervised Learning
python ncdia/train.py \
    --cfg configs/pipeline/supervised_learning/sl_in1k_rn18.yaml \
    --opts device='cuda:0'
