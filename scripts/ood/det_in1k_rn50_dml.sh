# Benchmark: ImageNet-1K
# Model: ResNet50
# Method: DML-Cosine
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/det_in1k_rn50_dml_cosine.yaml \
    --opts device='cuda:0'

# Model: ResNet50
# Method: DML-Norm
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/det_in1k_rn50_dml_norm.yaml \
    --opts device='cuda:0'

# Model: ResNet18
# Method: DML
# Task: Out-of-Distribution Detection
python train.py \
    --cfg configs/pipeline/ood_detection/det_in1k_rn18_dml.yaml \
    --opts device='cuda:0'
