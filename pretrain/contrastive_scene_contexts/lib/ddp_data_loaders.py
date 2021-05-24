# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import logging
import random
import torch
import torch.utils.data
import numpy as np
import glob
import os
import copy
from tqdm import tqdm
from scipy.linalg import expm, norm
from lib.io3d import write_triangle_mesh

import lib.transforms as t

import MinkowskiEngine as ME

from torch.utils.data.sampler import RandomSampler
from lib.data_sampler import DistributedInfSampler
import open3d as o3d


def make_open3d_point_cloud(xyz, color=None):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  if color is not None:
    pcd.colors = o3d.utility.Vector3dVector(color)
  return pcd


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
  source_copy = copy.deepcopy(source)
  target_copy = copy.deepcopy(target)
  source_copy.transform(trans)
  pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

  match_inds = []
  for i, point in enumerate(source_copy.points):
    [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
    if K is not None:
      idx = idx[:K]
    for j in idx:
      match_inds.append((i, j))
  return match_inds


def default_collate_pair_fn(list_data):
  xyz0, xyz1, coords0, coords1, feats0, feats1, label0, label1, instance0, instance1, matching_inds, trans, T0 = list(zip(*list_data))
  xyz_batch0, coords_batch0, feats_batch0, label_batch0, instance_batch0 = [], [], [], [], []
  xyz_batch1, coords_batch1, feats_batch1, label_batch1, instance_batch1 = [], [], [], [], []
  matching_inds_batch, trans_batch, len_batch, T0_batch = [], [], [], []

  batch_id = 0
  curr_start_inds = np.zeros((1, 2))
  for batch_id, _ in enumerate(coords0):

    N0 = coords0[batch_id].shape[0]
    N1 = coords1[batch_id].shape[0]

    # Move batchids to the beginning
    xyz_batch0.append(torch.from_numpy(xyz0[batch_id]))
    coords_batch0.append(
        torch.cat((torch.ones(N0, 1).float() * batch_id, 
                   torch.from_numpy(coords0[batch_id]).float()), 1))
    feats_batch0.append(torch.from_numpy(feats0[batch_id]))
    label_batch0.append(torch.from_numpy(label0[batch_id]))
    instance_batch0.append(torch.from_numpy(instance0[batch_id]))

    xyz_batch1.append(torch.from_numpy(xyz1[batch_id]))
    coords_batch1.append(
        torch.cat((torch.ones(N1, 1).float() * batch_id, 
                   torch.from_numpy(coords1[batch_id]).float()), 1))
    feats_batch1.append(torch.from_numpy(feats1[batch_id]))
    label_batch1.append(torch.from_numpy(label1[batch_id]))
    instance_batch1.append(torch.from_numpy(instance1[batch_id]))

    trans_batch.append(torch.from_numpy(trans[batch_id]))
    T0_batch.append(torch.from_numpy(T0[batch_id]))

    # in case 0 matching
    if len(matching_inds[batch_id]) == 0:
      matching_inds[batch_id].extend([0, 0])

    matching_inds_batch.append(
        torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
    len_batch.append([N0, N1])

    # Move the head
    curr_start_inds[0, 0] += N0
    curr_start_inds[0, 1] += N1

  # Concatenate all lists
  xyz_batch0 = torch.cat(xyz_batch0, 0).float()
  coords_batch0 = torch.cat(coords_batch0, 0).float()
  feats_batch0 = torch.cat(feats_batch0, 0).float()
  label_batch0 = torch.cat(label_batch0, 0).int()
  instance_batch0 = torch.cat(instance_batch0, 0).int()

  xyz_batch1 = torch.cat(xyz_batch1, 0).float()
  coords_batch1 = torch.cat(coords_batch1, 0).float()
  feats_batch1 = torch.cat(feats_batch1, 0).float()
  label_batch1 = torch.cat(label_batch1, 0).int()
  instance_batch1 = torch.cat(instance_batch1, 0).int()

  trans_batch = torch.cat(trans_batch, 0).float()
  T0_batch = torch.stack(T0_batch, 0).float()
  matching_inds_batch = torch.cat(matching_inds_batch, 0).int()

  return {
      'pcd0': xyz_batch0,
      'pcd1': xyz_batch1,
      'sinput0_C': coords_batch0,
      'sinput0_F': feats_batch0,
      'sinput0_L': label_batch0,
      'sinput0_I': instance_batch1,
      'sinput1_C': coords_batch1,
      'sinput1_F': feats_batch1,
      'sinput1_L': label_batch1,
      'sinput1_I': instance_batch1,
      'correspondences': matching_inds_batch,
      'trans': trans_batch,
      'T0': T0_batch,
      'len_batch': len_batch,
  }

# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

def sample_random_trans(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T

def sample_random_trans_z(pcd):
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
  rot_mats = []
  for axis_ind, rot_bound in enumerate(ROTATION_AUGMENTATION_BOUND):
    theta = 0
    axis = np.zeros(3)
    axis[axis_ind] = 1
    if rot_bound is not None:
      theta = np.random.uniform(*rot_bound)
    rot_mats.append(M(axis, theta))
  # Use random order
  np.random.shuffle(rot_mats)
  rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]

  T = np.eye(4)
  T[:3, :3] = rot_mat
  T[:3, 3] = rot_mat.dot(-np.mean(pcd, axis=0))
  return T

def only_trans(pcd):
  T = np.eye(4)
  T[:3, 3] = -np.mean(pcd, axis=0)
  return T

class PairDataset(torch.utils.data.Dataset):
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_scale=False,
               manual_seed=False,
               config=None):
    self.phase = phase
    self.files = []
    self.data_objects = []
    self.transform = transform
    self.voxel_size = config.data.voxel_size
    self.matching_search_voxel_size = \
        config.data.voxel_size * config.trainer.positive_pair_search_voxel_size_multiplier
    self.config = config

    self.random_scale = random_scale
    self.min_scale = 0.8
    self.max_scale = 1.2
    self.randg = np.random.RandomState()
    
    if manual_seed:
      self.reset_seed()
    
    self.root = '/'
    if phase == "train":
      self.root_filelist = root = config.data.scannet_match_dir
    else:
      raise NotImplementedError

    logging.info(f"Loading the subset {phase} from {root}")
    fname_txt = os.path.join(self.root_filelist, 'splits/overlap30.txt')
    with open(fname_txt) as f:
      content = f.readlines()
    fnames = [x.strip().split() for x in content]
    for fname in fnames:
      self.files.append([os.path.join(self.root_filelist, fname[0]), 
                         os.path.join(self.root_filelist, fname[1])])

  def reset_seed(self, seed=0):
    logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts
  
  def __len__(self):
    return len(self.files)

class ScanNetIndoorPairDataset(PairDataset):
  OVERLAP_RATIO = None
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_scale=False,
               manual_seed=False,
               config=None):
    PairDataset.__init__(self, phase, transform, random_scale, manual_seed, config)

    # add
    self.CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                  'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                  'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
    self.VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
    NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
    # 0-40
    IGNORE_LABELS = tuple(set(range(41)) - set(self.VALID_CLASS_IDS))
    self.label_map = {}
    n_used = 0
    for l in range(NUM_LABELS):
      if l in IGNORE_LABELS:
          self.label_map[l] = 255
      else:
          self.label_map[l] = n_used
          n_used += 1
    self.label_map[255] = 255

  def get_correspondences(self, idx):
    file0 = os.path.join(self.root, self.files[idx][0])
    file1 = os.path.join(self.root, self.files[idx][1])
    data0 = np.load(file0)
    data1 = np.load(file1)
    xyz0 = data0["pcd"][:,:3]
    xyz1 = data1["pcd"][:,:3]

    label0 = (data0["pcd"][:,6] / 1000).astype(np.int32)
    label1 = (data1["pcd"][:,6] / 1000).astype(np.int32)
    instance0 = (data0["pcd"][:,6] % 1000).astype(np.int32)
    instance1 = (data1["pcd"][:,6] % 1000).astype(np.int32)
    color0 = data0['pcd'][:,3:6] 
    color1 = data1['pcd'][:,3:6] 

    matching_search_voxel_size = self.matching_search_voxel_size

    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1
    
    if self.config.data.random_rotation_xyz:
      T0 = sample_random_trans(xyz0, self.randg)
      T1 = sample_random_trans(xyz1, self.randg)
    else:
      T0 = sample_random_trans_z(xyz0)
      T1 = sample_random_trans_z(xyz1)
    #else:
    #  T0 = only_trans(xyz0)
    #  T1 = only_trans(xyz1)

    trans = T1 @ np.linalg.inv(T0)
    xyz0 = self.apply_transform(xyz0, T0)
    xyz1 = self.apply_transform(xyz1, T1)

    # Voxelization
    sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size, return_index=True)

    if not self.config.data.voxelize:
      sel0 = sel0[np.random.choice(sel0.shape[0], self.config.data.num_points, 
                              replace=self.config.data.num_points>sel0.shape[0])]
      sel1 = sel1[np.random.choice(sel1.shape[0], self.config.data.num_points, 
                              replace=self.config.data.num_points>sel1.shape[0])]


    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0)
    pcd1 = make_open3d_point_cloud(xyz1)

    # Select features and points using the returned voxelized indices
    pcd0.colors = o3d.utility.Vector3dVector(color0[sel0])
    pcd1.colors = o3d.utility.Vector3dVector(color1[sel1])
    pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
    pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])
    label0 = label0[sel0]
    label1 = label1[sel1]
    color0 = color0[sel0]
    color1 = color1[sel1]
    instance0 = instance0[sel0]
    instance1 = instance1[sel1]
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)

    # Get features
    feats_train0, feats_train1 = [], []

    feats_train0.append(color0)
    feats_train1.append(color1)

    feats0 = np.hstack(feats_train0)
    feats1 = np.hstack(feats_train1)

    # Get coords
    xyz0 = np.array(pcd0.points)
    xyz1 = np.array(pcd1.points)

    if self.config.data.voxelize:
      coords0 = np.floor(xyz0 / self.voxel_size)
      coords1 = np.floor(xyz1 / self.voxel_size)
    else:
      coords0 = xyz0
      coords1 = xyz1

    #jitter color
    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    feats0 = feats0 / 255.0 - 0.5
    feats1 = feats1 / 255.0 - 0.5

    # label mapping for monitor
    label0 = np.array([self.label_map[x] for x in label0], dtype=np.int)
    label1 = np.array([self.label_map[x] for x in label1], dtype=np.int)

    # NB(s9xie): xyz are coordinates in the original system;
    # coords are sparse conv grid coords. (subject to a scaling factor)
    # coords0 -> sinput0_C
    # trans is T0*T1^-1
    return (xyz0, xyz1, coords0, coords1, feats0, feats1, label0, label1, instance0, instance1, matches, trans, T0)

  def __getitem__(self, idx):
    return self.get_correspondences(idx)

class ScanNetMatchPairDataset(ScanNetIndoorPairDataset):
  OVERLAP_RATIO = 0.3
  DATA_FILES = {
      'train': './config/train_scannet.txt',
  }


ALL_DATASETS = [ScanNetMatchPairDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, batch_size, num_threads=0):

  if config.data.dataset not in dataset_str_mapping.keys():
    logging.error(f'Dataset {config.data.dataset}, does not exists in ' +
                  ', '.join(dataset_str_mapping.keys()))

  Dataset = dataset_str_mapping[config.data.dataset]

  transforms = []
  transforms.append(t.Jitter())
  dset = Dataset(
      phase="train",
      transform=t.Compose(transforms),
      random_scale=False,
      config=config)

  collate_pair_fn = default_collate_pair_fn
  if config.misc.num_gpus > 1:
    sampler = DistributedInfSampler(dset)
  else:
    sampler = None
  
  loader = torch.utils.data.DataLoader(
      dset,
      batch_size=batch_size,
      shuffle=False if sampler else True,
      num_workers=num_threads,
      collate_fn=collate_pair_fn,
      pin_memory=False,
      sampler=sampler,
      drop_last=True)

  return loader
