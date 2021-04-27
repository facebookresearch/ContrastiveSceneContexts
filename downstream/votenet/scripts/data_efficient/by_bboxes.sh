#! /bin/bash

mkdir -p $LOG_DIR

# main script
python ddp_main.py -m \
  net.backbone=sparseconv \
  data.dataset=scannet \
  data.batch_size=32 \
  data.num_workers=4 \
  data.num_points=40000 \
  data.no_height=True \
  data.by_points=$SAMPLED_BBOX \
  optimizer.learning_rate=0.001 \
  optimizer.max_epoch=500 \
  data.voxelization=True \
  data.voxel_size=0.025 \
  misc.log_dir=$LOG_DIR \
  net.is_train=True \
  net.weights=$PRETRAIN \
