# Benchmark: OES
# Model: ResNet18
# Method: VOS
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/vos/det_oes_rn18_vos_train.yaml \
    --opts device='cuda:0'
