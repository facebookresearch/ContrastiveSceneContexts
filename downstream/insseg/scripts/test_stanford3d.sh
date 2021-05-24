#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


MODEL=Res16UNet34C

python ddp_main.py \
    train.is_train=False \
    train.lenient_weight_loading=True \
    net.model=${MODEL} \
    net.conv1_kernel_size=3 \
    net.weights=$PRETRAIN \
    data.dataset=StanfordArea5Dataset \
    data.voxel_size=0.05 \
    data.num_workers=1 \
    data.stanford3d_path=${DATAPATH} \
    augmentation.data_aug_color_trans_ratio=0.05 \
    augmentation.data_aug_color_jitter_std=0.005 \
    misc.log_dir=${LOG_DIR} \
    misc.train_stuff=True \
    hydra.launcher.partition=learnfair \
    hydra.launcher.comment=CVPR_Deadline \
