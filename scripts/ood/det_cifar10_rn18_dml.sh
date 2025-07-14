# Benchmark: CIFAR-10
# Model: ResNet18
# Method: DML-Cosine
# Task: Out-of-Distribution Detection
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/det_cifar10_rn18_dml_cosine.yaml \
    --opts device='cuda:0'

# Model: ResNet18
# Method: DML-Norm
# Task: Out-of-Distribution Detection
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/det_cifar10_rn18_dml_norm.yaml \
    --opts device='cuda:0'

# test DML with rn18
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/det_cifar10_rn18_dml.yaml \
    --opts device='cuda:0'
