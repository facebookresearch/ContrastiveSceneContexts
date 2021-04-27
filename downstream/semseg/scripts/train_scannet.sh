#!/bin/bash
MODEL=Res16UNet34C

python ddp_main.py \
    train.train_phase=train \
    train.val_phase=val \
    train.is_train=True \
    train.lenient_weight_loading=True \
    train.stat_freq=2 \
    train.val_freq=500 \
    net.model=${MODEL} \
    net.conv1_kernel_size=3 \
    augmentation.normalize_color=True \
    data.dataset=ScannetVoxelization2cmDataset \
    data.batch_size=48 \
    data.num_workers=2 \
    data.scannet_path=${DATAPATH} \
    data.return_transformation=False \
    optimizer.lr=0.8 \
    optimizer.scheduler=PolyLR \
    optimizer.max_iter=20000 \
    misc.log_dir=${LOG_DIR} \
    misc.num_gpus=8 \
    hydra.launcher.partition=dev \
    hydra.launcher.comment=CVPR_rebuttal \
    net.weights=$PRETRAIN \

