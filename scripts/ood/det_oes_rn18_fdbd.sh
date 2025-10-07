# Benchmark: OES
# Model: ResNet18
# Method: FDBD
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/fdbd/det_oes_rn18_fdbd_test.yaml \
    --opts device='cuda:0'
