# test
# export PYTHONPATH=/new_data/lyf/openhaiv:$PYTHONPATH
python ncdia/train.py \
    --cfg configs/pipeline/ood/det_oes_clip-b16_mls.yaml \
    --opts device='cuda:0'


