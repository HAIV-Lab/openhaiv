# Benchmark: OES
# Model: ResNet18
# Method: LogitNorm
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/det_oes_rn18_logitnorm.yaml \
    --opts device='cuda:0'
