#!/bin/bash

MODEL=Res16UNet34C

python ddp_main.py \
    train.is_train=False \
    train.lenient_weight_loading=True \
    net.model=${MODEL} \
    net.conv1_kernel_size=3 \
    net.weights=$PRETRAIN \
    data.dataset=StanfordArea5Dataset \
    data.voxel_size=0.05 \
    data.num_workers=2 \
    data.stanford3d_path=${DATAPATH} \
    augmentation.data_aug_color_trans_ratio=0.05 \
    augmentation.data_aug_color_jitter_std=0.005 \
    misc.log_dir=${LOG_DIR} \
    hydra.launcher.partition=learnfair \
    hydra.launcher.comment=CVPR_Deadline \
