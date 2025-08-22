# Benchmark: OES
# Model: ResNet50
# Method: ViM
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/vim/det_oes_rn50_vim_test.yaml \
    --opts device='cuda:0'
