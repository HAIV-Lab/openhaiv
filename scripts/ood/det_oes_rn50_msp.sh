# train
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/msp/det_oes_rn50_msp_train.yaml \
    --opts device='cuda:0'

# test
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/msp/det_oes_rn50_msp_test.yaml \
    --opts device='cuda:0'
