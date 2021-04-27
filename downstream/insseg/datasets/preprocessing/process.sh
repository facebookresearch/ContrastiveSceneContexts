#! /bin/bash

#python sampling_points.py --method kmeans --num_iters 10 --num_points 400 --xyz --rgb --output /checkpoint/jihou/data/scannet/sampled_inds/kmeansXYZRGB_iter10_points400
#python sampling_points.py --method supervised --num_points 20 --output ./supervised_points20
#python sampling_points.py --method random --output ./random_points50_effective25

#python sampling_points.py --method kmeans --num_iters 50 --num_points 20 --feat --xyz --output ./kmeansXYZFeat_iter50_points20
#python sampling_points.py --method kmeans --num_iters 50 --num_points 50 --feat --xyz --output ./kmeansXYZFeat_iter50_points50
#python sampling_points.py --method kmeans --num_iters 50 --num_points 100 --feat --xyz --output ./kmeansXYZFeat_iter50_points100
#python sampling_points.py --method kmeans --num_iters 50 --num_points 200 --feat --xyz --output ./kmeansXYZFeat_iter50_points200
#python sampling_points.py --method kmeans --num_iters 50 --num_points 400 --feat --xyz --output ./kmeansXYZFeats_iter50_points400

#python sampling_points.py --method kmeans --num_iters 50 --num_points 20 --feat --output ./kmeansFeat_iter50_points20
#python sampling_points.py --method kmeans --num_iters 50 --num_points 50 --feat --output ./kmeansFeat_iter50_points50
python sampling_points.py --method kmeans --num_iters 50 --num_points 100 --feat --output ./kmeansFeat_iter50_points100
#python sampling_points.py --method kmeans --num_iters 50 --num_points 200 --feat --output ./kmeansFeat_iter50_points200
python sampling_points.py --method kmeans --num_iters 50 --num_points 400 --feat --output ./kmeansFeats_iter50_points400



