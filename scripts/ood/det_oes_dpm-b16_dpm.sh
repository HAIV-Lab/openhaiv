# Benchmark: OES
# Model: CLIP-B/16
# SL Method: DPM
# OOD Method: DPM
# Task: Out-of-Distribution Detection
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/det_oes_dpm-b16_dpm.yaml \
    --opts device='cuda:0'
