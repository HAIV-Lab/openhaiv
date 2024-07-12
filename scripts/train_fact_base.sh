CUDA_VISIBLE_DEVICES=2 python main.py \
    --config configs/increment/fact_remote.yml \
    configs/dataloader/fact_remote.yml \
    configs/networks/resnet18_fact.yml \
    configs/pipelines/train_fact_base.yml \
    --seed 1