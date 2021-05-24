import logging
import os
import sys
from pathlib import Path

import torch
import numpy as np
from scipy import spatial

from datasets.dataset import VoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.utils import read_txt, fast_hist, per_class_iu
from lib.io3d import write_triangle_mesh, create_color_palette

class ScannetVoxelizationDataset(VoxelizationDataset):
  # added
  NUM_LABELS = 41  # Will be converted to 20 as defined in IGNORE_LABELS.
  NUM_IN_CHANNEL = 3
  CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
              'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
              'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
  VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
  IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))
  
  CLASS_LABELS_INSTANCE = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door',  'window', 'bookshelf', 'picture', 'counter',
                            'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
  VALID_CLASS_IDS_INSTANCE = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
  IGNORE_LABELS_INSTANCE = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS_INSTANCE))


  # Voxelization arguments
  CLIP_BOUND = None
  TEST_CLIP_BOUND = None
  VOXEL_SIZE = 0.05

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                        np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
  ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

  ROTATION_AXIS = 'z'
  LOCFEAT_IDX = 2
  IS_FULL_POINTCLOUD_EVAL = True

  # If trainval.txt does not exist, copy train.txt and add contents from val.txt
  DATA_PATH_FILE = {
      DatasetPhase.Train: 'scannetv2_train.txt',
      DatasetPhase.Val: 'scannetv2_val.txt',
      DatasetPhase.TrainVal: 'scannetv2_trainval.txt',
      DatasetPhase.Test: 'scannetv2_test.txt',
  }

  def __init__(self,
               config,
               prevoxel_transform=None,
               input_transform=None,
               target_transform=None,
               augment_data=True,
               elastic_distortion=False,
               cache=False,
               phase=DatasetPhase.Train):
    if isinstance(phase, str):
      phase = str2datasetphase_type(phase)
    # Use cropped rooms for train/val
    data_root = config.data.scannet_path
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    
    data_paths = read_txt(os.path.join(data_root, 'splits', self.DATA_PATH_FILE[phase]))
    if phase == DatasetPhase.Train and config.data.train_file:
      data_paths = read_txt(config.data.train_file)
    
    # data efficiency by sampling points
    self.sampled_inds = {}
    if config.data.sampled_inds and phase == DatasetPhase.Train:
      self.sampled_inds = torch.load(config.data.sampled_inds)

    data_paths = [data_path + '.pth' for data_path in data_paths]
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
    super().__init__(
        data_paths,
        data_root=data_root,
        prevoxel_transform=prevoxel_transform,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.data.ignore_label,
        return_transformation=config.data.return_transformation,
        augment_data=augment_data,
        elastic_distortion=elastic_distortion,
        config=config)

  def get_output_id(self, iteration):
    return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

  def _augment_locfeat(self, pointcloud):
    # Assuming that pointcloud is xyzrgb(...), append location feat.
    pointcloud = np.hstack(
        (pointcloud[:, :6], 100 * np.expand_dims(pointcloud[:, self.LOCFEAT_IDX], 1),
         pointcloud[:, 6:]))
    return pointcloud

  def load_data(self, index):
    filepath = self.data_root / self.data_paths[index]
    pointcloud = torch.load(filepath)
    coords = pointcloud[0].astype(np.float32)
    feats = pointcloud[1].astype(np.float32)
    labels = pointcloud[2].astype(np.int32)
    if self.sampled_inds:
      scene_name = self.get_output_id(index)
      mask = np.ones_like(labels).astype(np.bool)
      sampled_inds = self.sampled_inds[scene_name]
      mask[sampled_inds] = False
      labels[mask] = 0

    return coords, feats, labels
  
  def save_features(self, coords, upsampled_features, transformation, iteration, save_dir):
    inds_mapping, xyz = self.get_original_pointcloud(coords, transformation, iteration)
    ptc_feats = upsampled_features.cpu().numpy()[inds_mapping]
    room_id = self.get_output_id(iteration)
    torch.save(ptc_feats, f'{save_dir}/{room_id}')
  
  def get_original_pointcloud(self, coords, transformation, iteration):
    logging.info('===> Start testing on original pointcloud space.')
    data_path = self.data_paths[iteration]
    fullply_f = self.data_root / data_path
    query_xyz, _, query_label, _  = torch.load(fullply_f)

    coords = coords[:, 1:].numpy() + 0.5
    curr_transformation = transformation[0, :16].numpy().reshape(4, 4)
    coords = np.hstack((coords, np.ones((coords.shape[0], 1))))
    coords = (np.linalg.inv(curr_transformation) @ coords.T).T

    # Run test for each room.
    from pykeops.numpy import LazyTensor
    from pykeops.numpy.utils import IsGpuAvailable
    
    query_xyz = np.array(query_xyz)
    x_i = LazyTensor( query_xyz[:,None,:] )  # x_i.shape = (1e6, 1, 3)
    y_j = LazyTensor( coords[:,:3][None,:,:] )  # y_j.shape = ( 1, 2e6,3)
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M**2, N) symbolic matrix of squared distances
    indKNN = D_ij.argKmin(1, dim=1)  # Grid <-> Samples, (M**2, K) integer tensor
    inds = indKNN[:,0]
    return inds, query_xyz


class ScannetVoxelization2cmDataset(ScannetVoxelizationDataset):
  VOXEL_SIZE = 0.02
