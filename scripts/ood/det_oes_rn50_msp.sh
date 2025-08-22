# Benchmark: OES
# Model: ResNet50
# Method: MSP 
# Task: Out-of-Distribution Detection
# Training phase
python train.py \
    --cfg configs/pipeline/ood_detection/msp/det_oes_rn50_msp_train.yaml \
    --opts device='cuda:0'

# Testing phase
python train.py \
    --cfg configs/pipeline/ood_detection/msp/det_oes_rn50_msp_test.yaml \
    --opts device='cuda:0'
