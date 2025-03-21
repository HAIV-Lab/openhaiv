# train loss focal
python ncdia/train.py \
    --cfg configs/pipeline/ood/det_cifar10_rn18_dml.yaml \
    --opts device='cuda:0' model.loss=focal criterion.type=FocalLoss \
    work_dir=./output/supervised/sl_cifar10_rn18_DMLC

# train loss center
python ncdia/train.py \
    --cfg configs/pipeline/ood/det_cifar10_rn18_dml.yaml \
    --opts device='cuda:0' model.loss=center criterion.type=CenterLoss \
    criterion.num_classes=10 criterion.feat_dim=512 criterion.device='cuda:0' \
    work_dir=./output/supervised/sl_cifar10_rn18_DMLN

# test
# python ncdia/train.py \
#     --cfg configs/pipeline/ood/det_cifar10_rn18_dml.yaml \
#     --opts device='cuda:0' model.loss=dml trainer.max_epochs=0 \
#     work_dir=./output/supervised/sl_cifar10_rn18_DML \
#     model.checkpoint_C=output/supervised/sl_cifar10_rn18_DMLC/exp/latest.pth \
#     model.checkpoint_N=output/supervised/sl_cifar10_rn18_DMLN/exp/latest.pth
