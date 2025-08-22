# Benchmark: OES
# Model: CLIP-B/16
# SL Method: CoOp
# OOD Method: MLS
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/det_oes_coop-b16_mls.yaml \
    --opts device='cuda:0'
