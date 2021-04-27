#!/bin/bash
MODEL=Res16UNet34C

python ddp_main.py \
    train.train_phase=train \
    train.is_train=True \
    train.lenient_weight_loading=True \
    train.stat_freq=2 \
    train.val_freq=200 \
    net.model=${MODEL} \
    net.conv1_kernel_size=3 \
    augmentation.normalize_color=True \
    data.dataset=ScannetVoxelization2cmDataset \
    data.batch_size=48 \
    data.sampled_inds=$SAMPLED_INDS \
    data.num_workers=1 \
    data.num_val_workers=1 \
    data.scannet_path=${DATAPATH} \
    data.return_transformation=False \
    optimizer.lr=0.1 \
    optimizer.scheduler=PolyLR \
    optimizer.max_iter=10000 \
    misc.log_dir=${LOG_DIR} \
    misc.num_gpus=8 \
    hydra.launcher.partition=priority \
    hydra.launcher.comment=CVPR_Deadline \
    net.weights=$PRETRAIN \

