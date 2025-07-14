# Benchmark: CIFAR-10
# Model: ResNet18
# Method: MLS
# Task: Out-of-Distribution Detection
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/det_cifar10_rn18_mls.yaml \
    --opts device='cuda:0'
