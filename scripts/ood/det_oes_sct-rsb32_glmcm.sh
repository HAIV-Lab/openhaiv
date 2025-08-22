# Benchmark: OES
# Model: RSCLIP-B/32
# SL Method: SCT
# OOD Method: GL-MCM
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/det_oes_sct-rsb32_glmcm.yaml \
    --opts device='cuda:0'
