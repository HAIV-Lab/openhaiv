# Benchmark: OES
# Model: ResNet18
# Method: Energy
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/energy/det_oes_rn18_energy_test.yaml \
    --opts device='cuda:0'
