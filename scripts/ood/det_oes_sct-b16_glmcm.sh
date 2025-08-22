# Benchmark: OES
# Model: CLIP-B/16
# SL Method: SCT
# OOD Method: GL-MCM
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/det_oes_sct-b16_glmcm.yaml \
    --opts device='cuda:0'
