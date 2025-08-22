# Benchmark: OES
# Model: ResNet50
# Method: LogitNorm
# Task: Out-of-Distribution Detection
# Training phase
python train.py \
    --cfg configs/pipeline/ood_detection/logitnorm/det_oes_rn50_logitnorm_train.yaml \
    --opts device='cuda:0'
# Testing phase  
python train.py \
    --cfg configs/pipeline/ood_detection/logitnorm/det_oes_rn50_logitnorm_test.yaml \
    --opts device='cuda:0'

