# Benchmark: OES
# Model: ResNet50
# Method: MSP 
# Task: Out-of-Distribution Detection
# lr=0.001
python train.py \
    --cfg configs/pipeline/ood_detection/msp/det_oes_rn50_msp_train_lr0.001.yaml \
    --opts device='cuda:0'

# lr=0.02
python train.py \
    --cfg configs/pipeline/ood_detection/msp/det_oes_rn50_msp_train_lr0.02.yaml \
    --opts device='cuda:0'

# lr=0.05
python train.py \
    --cfg configs/pipeline/ood_detection/msp/det_oes_rn50_msp_train_lr0.05.yaml \
    --opts device='cuda:0'

# lr=0.1
python train.py \
    --cfg configs/pipeline/ood_detection/msp/det_oes_rn50_msp_train_lr0.1.yaml \
    --opts device='cuda:0'

# lr=0.002 
python train.py \
    --cfg configs/pipeline/ood_detection/msp/det_oes_rn50_msp_train_lr0.002.yaml \
    --opts device='cuda:0'

# lr=0.005
python train.py \
    --cfg configs/pipeline/ood_detection/msp/det_oes_rn50_msp_train_lr0.005.yaml \
    --opts device='cuda:0'


# lr=0.01
python train.py \
    --cfg configs/pipeline/ood_detection/msp/det_oes_rn50_msp_train_lr0.01.yaml \
    --opts device='cuda:0'
