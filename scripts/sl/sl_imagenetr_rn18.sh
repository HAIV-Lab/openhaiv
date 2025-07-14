# Benchmark: Imagenet-R
# Model: ResNet18
# Method: Cross-Entropy
# Task: Supervised Learning
python ncdia/train.py \
    --cfg configs/pipeline/supervised_learning/sl_imagenetr_rn18.yaml \
    --opts device='cuda:0'
