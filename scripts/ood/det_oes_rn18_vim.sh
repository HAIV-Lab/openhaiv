# Benchmark: OES
# Model: ResNet18
# Method: ViM
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/vim/det_oes_rn18_vim_test.yaml \
    --opts device='cuda:0'
