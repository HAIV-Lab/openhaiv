python main.py \
    --config configs/increment/savc_remote_IR.yml \
    configs/dataloader/savc_remote_IR.yml \
    configs/networks/resnet18_savc_IR.yml \
    configs/pipelines/train_savc_new_IR.yml \
    --seed 4 \
    --CIL.shot 13 \
    --discoverer.name 'savc_sskm_discoverer' \
    --discoverer.sift_threshold 0.0 \
    --network.checkpoint '/new_data/dx450/Project/zzh_ood/NCD_Attr_zzh/results_IR/remote_savc_att_base_resnet18_savc_att_q_savc_att_e10_lr0.002_default/s1/ReFc_Final_acc0.9686.ckpt' \
    --dataloader.split_file_test '/new_data/dx450/IRBenchmark/test/' \
    --dataloader.txt_path '/new_data/dx450/Project/zzh_ood/NCD_Attr_zzh/data/index_list_ir_sskm/' \
    --quantizer.apply 'False'