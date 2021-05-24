# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#! /bin/bash

mkdir -p $LOG_DIR

# main script
python ddp_main.py \
  net.is_train=False \
  net.backbone=sparseconv \
  data.dataset=sunrgbd \
  data.num_workers=4 \
  data.num_points=20000 \
  data.no_height=True \
  data.voxelization=True \
  data.voxel_size=0.025 \
  test.ap_iou=0.5 \
  optimizer.learning_rate=0.001 \
  misc.log_dir=$LOG_DIR \
  net.weights=$PRETRAIN \
