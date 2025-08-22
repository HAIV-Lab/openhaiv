# Benchmark: OES
# Model: RSCLIP-B/32
# SL Method: CoOp
# OOD Method: MCM
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/det_oes_coop-rsb32_mcm.yaml \
    --opts device='cuda:0'
