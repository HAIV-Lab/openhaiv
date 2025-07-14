# Benchmark: OES
# Model: CLIP-B/16
# Method: SCT
# Task: Supervised Learning
python ncdia/train.py \
    --cfg configs/pipeline/supervised_learning/sl_oes_sct-b16.yaml \
    --opts device='cuda:0'
