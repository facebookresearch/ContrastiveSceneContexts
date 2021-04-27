import os
import torch
import numpy as np
import glob
import time
import argparse
import pykeops
from pykeops.torch import LazyTensor
pykeops.clean_pykeops() 

def parse_args():
    """parse input arguments"""
    parser = argparse.ArgumentParser(description='data_efficient3d')
    parser.add_argument('--raw_data', type=str, default='/checkpoint/jihou/data/scannet/pointcloud/')
    parser.add_argument('--feat_data', type=str, default='/checkpoint/jihou/checkpoint/scannet/pretrain/partition8_4096_60k/1/outputs/feat')
    #parser.add_argument('--feat_data', type=str, default='/checkpoint/jihou/checkpoint/scannet/pretrain/pointcontrastNCE/0/outputs/feat')

    parser.add_argument('--xyz', action='store_true')
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--feat', action='store_true')

    parser.add_argument('--num_points', type=int, default=100)
    parser.add_argument('--num_iters', type=int, default=10)
    parser.add_argument('--method', type=str, default='kmeans', help='random, kmeans, supervised')
    parser.add_argument('--output', type=str, default='/rdata/ji/data/scannet/sampled_inds_kmeans_100_feats')
    return parser.parse_args()


def get_valid_inds(labels):
    VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
    valid_inds = []
    for label in VALID_CLASS_IDS:
        valid_inds.append((labels == label).nonzero()[0])
    valid_inds = np.concatenate(valid_inds)
    return valid_inds

def supervised_sampling(labels, instances, valid_inds, k):
    instance_uniques = np.unique(instances[valid_inds])
    sampled_inds = []
    for instance in instance_uniques:
        num = round(k / len(instance_uniques))
        inds = (instances == instance).nonzero()[0]
        inds = np.random.choice(inds, num)
        sampled_inds.append(inds)
    return np.concatenate(sampled_inds)

def kmeans(pointcloud, k=10, iterations=10, verbose=True):
    n, dim = pointcloud.shape  # Number of samples, dimension of the ambient space

    start = time.time()
    clusters = pointcloud[:k, :].clone()  # Simplistic random initialization
    pointcloud_cuda = LazyTensor(pointcloud[:, None, :])  # (Npoints, 1, D)

    # K-means loop:
    for _ in range(iterations):
        clusters_previous = clusters.clone()
        clusters_gpu = LazyTensor(clusters[None, :, :])  # (1, Nclusters, D)
        distance_matrix = ((pointcloud_cuda - clusters_gpu) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cloest_clusters = distance_matrix.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # #points for each cluster
        clusters_count = torch.bincount(cloest_clusters, minlength=k).float()  # Class weights
        for d in range(dim):  # Compute the cluster centroids with torch.bincount:
            clusters[:, d] = torch.bincount(cloest_clusters, weights=pointcloud[:, d], minlength=k) / clusters_count
        
        # for clusters that have no points assigned
        mask = clusters_count == 0
        clusters[mask] = clusters_previous[mask]

    end = time.time()

    if verbose:
        print("K-means example with {:,} points in dimension {:,}, K = {:,}:".format(n, dim, k))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
                iterations, end - start, iterations, (end-start) / iterations))
    
    # nearest neighbouring search for each cluster
    cloest_points_to_centers = distance_matrix.argmin(dim=0).long().view(-1)
    return cloest_points_to_centers

def sampling(args):
    pointcloud_names = glob.glob(os.path.join(args.raw_data, "*.pth"))

    sampled_inds = {}
    for idx, pointcloud_name in enumerate(pointcloud_names):
        print('{}/{}: {}'.format(idx, len(pointcloud_names), pointcloud_name))
        pointcloud = torch.load(pointcloud_name)
        scene_name = os.path.basename(pointcloud_name).split('.')[0]

        coords = pointcloud[0].astype(np.float32)
        colors = pointcloud[1].astype(np.int32)
        labels = pointcloud[2].astype(np.int32)
        instances = pointcloud[3].astype(np.int32)
        num_points = coords.shape[0]

        candidates = []
        if args.xyz:
            candidates.append(coords)
        if args.rgb:
            candidates.append(colors)
        if args.feat:
            try:
                feats = torch.load(os.path.join(args.feat_data, scene_name))
            except:
                print('{} not exists'.format(scene_name))
                continue
            candidates.append(feats)
        candidates = torch.from_numpy(np.concatenate(candidates,1)).cuda().float()

        K = args.num_points
        if args.method == 'kmeans':
            sampled_inds_per_scene = kmeans(candidates, K, args.num_iters).cpu().numpy()
        elif args.method == 'random':
            sampled_inds_per_scene = np.random.choice(num_points, K)
        elif args.method == 'supervised':
            valid_inds = get_valid_inds(labels)
            if len(valid_inds) == 0:
                valid_inds = np.setdiff1d(np.arange(num_points), valid_inds)
            sampled_inds_per_scene = supervised_sampling(labels, instances, valid_inds, K)

        sampled_inds[scene_name] = sampled_inds_per_scene


    return sampled_inds


if __name__ == "__main__":
    args = parse_args()
    sampled_inds = sampling(args)
    torch.save(sampled_inds, args.output)
