# Benchmark: OES
# Model: ResNet50
# Method: GODIN
# Task: Out-of-Distribution Detection
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/godin/det_oes_rn50_godin_train.yaml \
    --opts device='cuda:1'
# Test phase    
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/godin/det_oes_rn50_godin_test.yaml \
    --opts device='cuda:1'
