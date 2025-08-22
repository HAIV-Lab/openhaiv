# Benchmark: Food101
# Model: ResNet18
# Method: Cross-Entropy
# Task: Supervised Learning
python train.py \
    --cfg configs/pipeline/supervised_learning/sl_food101_rn18.yaml \
    --opts device='cuda:0'
