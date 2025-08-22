# Benchmark: OES
# Model: RSCLIP-B/32
# Method: DPM
# Task: Supervised Learning
python train.py \
    --cfg configs/pipeline/supervised_learning/sl_oes_dpm-rsb32.yaml \
    --opts device='cuda:0'
