# Benchmark: OES
# Model: ResNet18
# Method: MSP
# Task: Out-of-Distribution Detection
# Training & Testing
python train.py \
    --cfg configs/pipeline/ood_detection/msp/det_oes_rn18_msp_train.yaml \
    --opts device='cuda:0'
