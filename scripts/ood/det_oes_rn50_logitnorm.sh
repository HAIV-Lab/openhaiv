# test
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/logitnorm/det_oes_rn50_logitnorm_train.yaml \
    --opts device='cuda:0'
    
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/logitnorm/det_oes_rn50_logitnorm_test.yaml \
    --opts device='cuda:0'

