# Benchmark: OES
# Model: ResNet50
# Method: Cross-Entropy
# Task: Supervised Learning
python train.py \
    --cfg configs/pipeline/supervised_learning/sl_oes_rn50.yaml \
    --opts device='cuda:0'
