# train loss focal
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/det_in1k_rn50_dml_cosine.yaml \
    --opts device='cuda:0'

# train loss center
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/det_in1k_rn50_dml_norm.yaml \
    --opts device='cuda:0'

# test
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/det_in1k_rn18_dml.yaml \
    --opts device='cuda:0'
