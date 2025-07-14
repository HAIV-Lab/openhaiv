# Benchmark: OES
# Model: CLIP-B/16
# Method: GL-MCM
# Task: Out-of-Distribution Detection
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/det_oes_clip-b16_glmcm.yaml \
    --opts device='cuda:0'


