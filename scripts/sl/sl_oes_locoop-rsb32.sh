# Benchmark: OES
# Model: RSCLIP-B/32
# Method: LoCoOp
# Task: Supervised Learning
python train.py \
    --cfg configs/pipeline/supervised_learning/sl_oes_locoop-rsb32.yaml \
    --opts device='cuda:0'
