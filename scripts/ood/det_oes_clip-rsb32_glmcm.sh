# test
# export PYTHONPATH=/new_data/lyf/openhaiv:$PYTHONPATH
python ncdia/train.py \
    --cfg configs/pipeline/ood_detection/det_oes_clip-rsb32_glmcm.yaml \
    --opts device='cuda:0'


