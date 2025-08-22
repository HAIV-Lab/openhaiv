# Benchmark: OES
# Model: ResNet50
# Method: MLS
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/mls/det_oes_rn50_mls_test.yaml \
    --opts device='cuda:1'
