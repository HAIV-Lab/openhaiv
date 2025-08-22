# Benchmark: OES
# Model: RSCLIP-B/32
# SL Method: DPM
# OOD Method: DPM
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/det_oes_dpm-rsb32_dpm.yaml \
    --opts device='cuda:0'
