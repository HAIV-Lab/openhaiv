# test
# export PYTHONPATH=/new_data/lyf/openhaiv:$PYTHONPATH
python ncdia/train.py \
    --cfg configs/pipeline/ood/det_oes_clip-rsb32_msp.yaml \
    --opts device='cuda:0'


