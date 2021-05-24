# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

export SCANNET_DIR=<path_to_downloaded_data>
export TARGET=<path_to_target_data>   # data destination (change here)

reader() {
    filename=$1
    frame_skip=25

    scene=$(basename -- "$filename")
    scene="${scene%.*}"
    echo "Find sens data: $filename $scene"
    python -u reader.py --filename $filename --output_path $TARGET/$scene --frame_skip $frame_skip --export_depth_images --export_poses --export_intrinsics
    echo "Extract point-cloud data"
    python -u point_cloud_extractor.py --input_path $TARGET/$scene --output_path $TARGET/$scene/pcd --save_npz
    echo "Compute partial scan overlapping"
    python -u compute_full_overlapping.py --input_path $TARGET/$scene/pcd
}

export -f reader

parallel -j 64 --linebuffer time reader ::: `find $SCANNET_DIR/scans/scene*/*.sens`
