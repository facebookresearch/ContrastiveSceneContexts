#! /bin/bash


mkdir -p $LOG_DIR

# main script
python ddp_main.py \
  net.backbone=pointnet2 \
  data.dataset=scannet \
  data.num_workers=4 \
  data.num_points=40000 \
  data.use_color=False \
  data.no_height=True \
  optimizer.learning_rate=0.001 \
  data.voxelization=False \
  misc.log_dir=$LOG_DIR \
  misc.num_gpus=1 \
  net.is_train=False \
  net.weights=$PRETRAIN \
