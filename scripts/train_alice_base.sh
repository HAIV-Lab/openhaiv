CUDA_VISIBLE_DEVICES=2 python main.py \
    --config configs/increment/alice_remote.yml \
    configs/dataloader/alice_remote.yml \
    configs/networks/resnet18_alice.yml \
    configs/pipelines/train_alice_base.yml \
    --seed 1