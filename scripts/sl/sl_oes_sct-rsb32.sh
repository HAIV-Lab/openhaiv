# Benchmark: OES
# Model: RSCLIP-B/32
# Method: SCT
# Task: Supervised Learning
python ncdia/train.py \
    --cfg configs/pipeline/supervised_learning/sl_oes_sct-rsb32.yaml \
    --opts device='cuda:0'
