#!/bin/bash
MODEL=Res16UNet34C

python ddp_main.py \
    train.is_train=False \
    train.lenient_weight_loading=True \
    net.model=${MODEL} \
    net.weights=$PRETRAIN \
    net.conv1_kernel_size=3 \
    augmentation.normalize_color=True \
    data.dataset=ScannetVoxelization2cmDataset \
    data.num_workers=1 \
    data.scannet_path=${DATAPATH} \
    data.return_transformation=True \
    test.save_features=True \
    test.test_phase=train \
    misc.log_dir=${LOG_DIR} \
