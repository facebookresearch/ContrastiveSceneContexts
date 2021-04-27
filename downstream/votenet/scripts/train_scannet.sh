#! /bin/bash

mkdir -p $LOG_DIR

# main script
python ddp_main.py -m \
  net.backbone=sparseconv \
  data.dataset=scannet \
  data.num_workers=8 \
  data.batch_size=32 \
  data.num_points=40000 \
  data.use_color=False \
  data.no_height=True \
  optimizer.learning_rate=0.001 \
  data.voxelization=True \
  data.voxel_size=0.025 \
  misc.log_dir=$LOG_DIR \
  misc.num_gpus=1 \
  net.is_train=True \
  test.ap_iou=0.5 \
  net.weights=$PRETRAIN \
