#! /bin/bash

mkdir -p $LOG_DIR

# main script
python ddp_main.py -m \
  net.backbone=pointnet2 \
  data.dataset=scannet \
  data.num_workers=4 \
  data.batch_size=8 \
  data.num_points=40000 \
  data.use_color=False \
  data.no_height=True \
  optimizer.learning_rate=0.001 \
  data.voxelization=False \
  data.voxel_size=0.025 \
  misc.log_dir=$LOG_DIR \
  misc.num_gpus=1 \
  test.ap_iou=0.5 \
  net.is_train=True \
  net.weights=$PRETRAIN \
