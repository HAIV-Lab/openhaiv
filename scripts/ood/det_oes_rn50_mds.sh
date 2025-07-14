# Benchmark: OES
# Model: ResNet50
# Method: MDS
# Task: Out-of-Distribution Detection
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/mds/det_oes_rn50_mds_test.yaml \
    --opts device='cuda:0'
