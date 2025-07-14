# Benchmark: OES
# Model: ResNet50
# Method: ODIN
# Task: Out-of-Distribution Detection
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/odin/det_oes_rn50_odin_test.yaml \
    --opts device='cuda:0'
