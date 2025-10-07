# Benchmark: OES
# Model: ResNet18
# Method: GODIN
# Task: Out-of-Distribution Detection
# Training phase
python train.py \
    --cfg configs/pipeline/ood_detection/godin/det_oes_rn18_godin_train.yaml \
    --opts device='cuda:0'
    
# Testing phase    
# python train.py \
#     --cfg configs/pipeline/ood_detection/godin/det_oes_rn18_godin_test.yaml \
#     --opts device='cuda:0'
