# Benchmark: OES
# Model: RSCLIP-B/32
# Method: MCM
# Task: Out-of-Distribution Detection
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/det_oes_clip-rsb32_mcm.yaml \
    --opts device='cuda:0'


