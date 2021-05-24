#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

MODEL=Res16UNet34C

python ddp_main.py \
    train.train_phase=train \
    train.is_train=True \
    train.lenient_weight_loading=True \
    train.stat_freq=5 \
    train.val_freq=200 \
    train.overwrite_weights=False \
    net.model=${MODEL} \
    net.conv1_kernel_size=3 \
    data.dataset=StanfordArea5Dataset \
    data.batch_size=48 \
    data.voxel_size=0.05 \
    data.num_workers=2 \
    data.stanford3d_path=${DATAPATH} \
    augmentation.data_aug_color_trans_ratio=0.05 \
    augmentation.data_aug_color_jitter_std=0.005 \
    optimizer.lr=0.1 \
    optimizer.scheduler=PolyLR \
    optimizer.max_iter=15000 \
    misc.log_dir=${LOG_DIR} \
    misc.num_gpus=8 \
    misc.train_stuff=True \
    hydra.launcher.partition=priority \
    hydra.launcher.comment=CriticalEXP \
    net.weights=$PRETRAIN \
