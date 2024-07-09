CUDA_VISIBLE_DEVICES=1 python main.py \
    --config configs/increment/savc_remote_IR.yml \
    configs/dataloader/savc_remote_IR.yml \
    configs/networks/resnet18_savc_IR.yml \
    configs/pipelines/train_savc_base_IR.yml \
    --seed 10 \
    # --dataloader.split_file_test "/new_data/dx450/IRBenchmark/test/"