# Benchmark: OES
# Model: ResNet50
# Method: FDBD
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/fdbd/det_oes_rn50_fdbd_test.yaml \
    --opts device='cuda:0'
