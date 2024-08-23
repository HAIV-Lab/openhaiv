CUDA_VISIBLE_DEVICES=2 python main.py \
    --config configs/increment/savc_remote_SAR.yml \
    configs/dataloader/savc_remote_SAR.yml \
    configs/networks/resnet18_savc_SAR.yml \
    configs/pipelines/train_savc_base_SAR.yml \
    --seed 1