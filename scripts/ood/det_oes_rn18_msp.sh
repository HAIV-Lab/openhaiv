# Benchmark: OES
# Model: ResNet18
# Method: MSP
# Task: Out-of-Distribution Detection
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/det_oes_rn18_msp.yaml \
    --opts device='cuda:0'
