#!/bin/bash
MODEL=Res16UNet34C

python ddp_main.py \
    train.train_phase=train \
    train.val_phase=val \
    train.is_train=True \
    train.lenient_weight_loading=True \
    train.stat_freq=2 \
    train.val_freq=200 \
    net.model=${MODEL} \
    net.conv1_kernel_size=3 \
    augmentation.normalize_color=True \
    data.dataset=ScannetVoxelization2cmDataset \
    data.train_file=$TRAIN_FILE \
    data.batch_size=6 \
    data.num_workers=2 \
    data.scannet_path=${DATAPATH} \
    data.return_transformation=False \
    optimizer.lr=0.8 \
    optimizer.scheduler=PolyLR \
    optimizer.max_iter=5000 \
    misc.log_dir=${LOG_DIR} \
    misc.num_gpus=1 \
    hydra.launcher.partition=learnfair \
    hydra.launcher.comment=CVPR_Deadline \
    net.weights=$PRETRAIN \

