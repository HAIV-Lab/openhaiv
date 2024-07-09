# GPU=1
# CPU=1
# node=73
# jobname=ncdia

# PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
CUDA_VISIBLE_DEVICES=2 python main.py \
    --config configs/increment/savc_remote.yml \
    configs/dataloader/savc_remote.yml \
    configs/networks/resnet18_savc.yml \
    configs/pipelines/train_savc_base.yml \
    --seed 1