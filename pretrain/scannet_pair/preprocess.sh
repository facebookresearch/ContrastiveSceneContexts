reader() {
    TARGET="/rhome/jhou/data/dataset/scannet/partial_frames"   # data destination (change here)
    filename=$1
    frame_skip=25

    scene=$(basename -- "$filename")
    scene="${scene%.*}"
    #echo "Find sens data: $filename $scene"
    #python -u reader.py --filename $filename --output_path $TARGET/$scene --frame_skip $frame_skip --export_depth_images --export_poses --export_intrinsics
    echo "Extract point-cloud data"
    python -u point_cloud_extractor.py --input_path $TARGET/$scene --output_path $TARGET/$scene/pcd --save_npz
    #echo "Compute partial scan overlapping"
    #python -u compute_full_overlapping.py --input_path $TARGET/$scene/pcd
}



export -f reader

parallel -j 1 --linebuffer time reader ::: `find  /canis/Datasets/ScanNet/public/v1/scans/scene*/*.sens`
