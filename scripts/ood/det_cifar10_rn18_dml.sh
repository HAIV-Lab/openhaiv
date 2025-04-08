set -e

# train loss focal
python ncdia/train.py \
    --cfg configs/pipeline/ood/det_cifar10_rn18_dml_cosine.yaml \
    --opts device='cuda:0'

# train loss center
python ncdia/train.py \
    --cfg configs/pipeline/ood/det_cifar10_rn18_dml_norm.yaml \
    --opts device='cuda:0'

# test
python ncdia/train.py \
    --cfg configs/pipeline/ood/det_cifar10_rn18_dml.yaml \
    --opts device='cuda:0'
