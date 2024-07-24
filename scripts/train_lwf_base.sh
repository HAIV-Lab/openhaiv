CUDA_VISIBLE_DEVICES=2 python main.py \
    --config configs/increment/lwf_remote.yml \
    configs/dataloader/lwf_remote.yml \
    configs/networks/resnet18_lwf.yml \
    configs/pipelines/train_lwf_base.yml \
    --seed 1