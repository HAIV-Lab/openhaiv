CUDA_VISIBLE_DEVICES=1
python ncdia/train.py --cfg configs/pipeline/incremental_leanring/inc_cub_bic.yaml --opts device='cuda:0'