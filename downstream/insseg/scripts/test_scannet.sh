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
    augmentation.normalize_color=True \
    data.dataset=ScannetVoxelization2cmDataset \
    data.num_workers=1 \
    data.scannet_path=${DATAPATH} \
    data.return_transformation=False \
    test.test_phase=val \
    test.evaluate_benchmark=False \
    misc.log_dir=${LOG_DIR} \
    misc.num_gpus=1 \
    net.weights=$PRETRAIN \

