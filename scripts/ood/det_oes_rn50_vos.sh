# Benchmark: OES
# Model: ResNet50
# Method: VOS
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/vos/det_oes_rn50_vos_test.yaml \
    --opts device='cuda:1'
