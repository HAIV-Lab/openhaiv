# Benchmark: OES
# Model: RSCLIP-B/32
# SL Method: LoCoOp
# OOD Method: GL-MCM
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/det_oes_locoop-rsb32_glmcm.yaml \
    --opts device='cuda:0'
