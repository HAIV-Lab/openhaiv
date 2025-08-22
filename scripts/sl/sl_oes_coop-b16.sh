# Benchmark: OES
# Model: CLIP-B/16
# Method: CoOp
# Task: Supervised Learning
python train.py \
    --cfg configs/pipeline/supervised_learning/sl_oes_coop-b16.yaml \
    --opts device='cuda:0'
