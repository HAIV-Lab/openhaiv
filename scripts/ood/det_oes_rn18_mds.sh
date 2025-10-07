# Benchmark: OES
# Model: ResNet18
# Method: MDS
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/mds/det_oes_rn18_mds_test.yaml \
    --opts device='cuda:0'
