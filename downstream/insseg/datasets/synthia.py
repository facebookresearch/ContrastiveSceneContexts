# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import unittest
import imageio
import os
import os.path as osp
import pickle
import numpy as np

from collections import defaultdict
from plyfile import PlyData

from lib.pc_utils import Camera, read_plyfile
from lib.dataset import DictDataset, VoxelizationDataset, TemporalVoxelizationDataset, \
    str2datasetphase_type, DatasetPhase
from lib.transforms import cfl_collate_fn_factory
from lib.utils import read_txt, debug_on


class SynthiaDataset(DictDataset):
  NUM_LABELS = 16

  def __init__(self, data_path_file, input_transform=None, target_transform=None):
    with open(data_path_file, 'r') as f:
      data_paths = pickle.load(f)
    super(SynthiaDataset, self).__init__(data_paths, input_transform, target_transform)

  @staticmethod
  def load_extrinsics(extrinsics_file):
    """Load the camera extrinsics from a .txt file.
    """
    lines = read_txt(extrinsics_file)
    params = [float(x) for x in lines[0].split(' ')]
    extrinsics_matrix = np.asarray(params).reshape([4, 4])
    return extrinsics_matrix

  @staticmethod
  def load_intrinsics(intrinsics_file):
    """Load the camera intrinsics from a intrinsics.txt file.

    intrinsics.txt: a text file containing 4 values that represent (in this order) {focal length,
                    principal-point-x, principal-point-y, baseline (m) with the corresponding right
                    camera}
    """
    lines = read_txt(intrinsics_file)
    assert len(lines) == 7
    intrinsics = {
        'focal_length': float(lines[0]),
        'pp_x': float(lines[2]),
        'pp_y': float(lines[4]),
        'baseline': float(lines[6]),
    }
    return intrinsics

  @staticmethod
  def load_depth(depth_file):
    """Read a single depth map (.png) file.

    1280x760
    760 rows, 1280 columns.
    Depth is encoded in any of the 3 channels in centimetres as an ushort.
    """
    img = np.asarray(imageio.imread(depth_file, format='PNG-FI'))  # uint16
    img = img.astype(np.int32)  # Convert to int32 for torch compatibility
    return img

  @staticmethod
  def load_label(label_file):
    """Load the ground truth semantic segmentation label.

    Annotations are given in two channels. The first channel contains the class of that pixel
    (see the table below). The second channel contains the unique ID of the instance for those
    objects that are dynamic (cars, pedestrians, etc.).

    Class         R       G       B       ID

    Void          0       0       0       0
    Sky             128   128     128     1
    Building        128   0       0       2
    Road            128   64      128     3
    Sidewalk        0     0       192     4
    Fence           64    64      128     5
    Vegetation      128   128     0       6
    Pole            192   192     128     7
    Car             64    0       128     8
    Traffic Sign    192   128     128     9
    Pedestrian      64    64      0       10
    Bicycle         0     128     192     11
    Lanemarking   0       172     0       12
    Reserved      -       -       -       13
    Reserved      -       -       -       14
    Traffic Light 0       128     128     15
    """
    img = np.asarray(imageio.imread(label_file, format='PNG-FI'))  # uint16
    img = img.astype(np.int32)  # Convert to int32 for torch compatibility
    return img

  @staticmethod
  def load_rgb(rgb_file):
    """Load RGB images. 1280x760 RGB images used for training.

    760 rows, 1280 columns.
    """
    img = np.array(imageio.imread(rgb_file))  # uint8
    return img


class SynthiaVoxelizationDataset(VoxelizationDataset):
  """Load the ground truth semantic segmentation label.
  Annotations are given in two channels. The first channel contains the class of that pixel
  (see the table below). The second channel contains the unique ID of the instance for those
  objects that are dynamic (cars, pedestrians, etc.).
  Class         R       G       B       ID
  Void          0       0       0       0
  Sky             128   128     128     1
  Building        128   0       0       2
  Road            128   64      128     3
  Sidewalk        0     0       192     4
  Fence           64    64      128     5
  Vegetation      128   128     0       6
  Pole            192   192     128     7
  Car             64    0       128     8
  Traffic Sign    192   128     128     9
  Pedestrian      64    64      0       10
  Bicycle         0     128     192     11
  Lanemarking   0       172     0       12
  Reserved      -       -       -       13
  Reserved      -       -       -       14
  Traffic Light 0       128     128     15
  """

  
  CLASS_LABELS = ('building', 'road', 'sidewalk', 'fence', 'vegetation', 'pole', 'car',
              'sign', 'pedestrian', 'cyclist', 'lanemarking', 'traffic light')
  VALID_CLASS_IDS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15)

  # Voxelization arguments
  CLIP_BOUND = ((-1800, 1800), (-1800, 1800), (-1800, 1800))
  TEST_CLIP_BOUND = ((-2500, 2500), (-2500, 2500), (-2500, 2500))
  VOXEL_SIZE = 15  # cm

  PREVOXELIZATION_VOXEL_SIZE = 7.5
  # Elastic distortion, (granularity, magitude) pairs
  # ELASTIC_DISTORT_PARAMS = ((80, 300),)

  # Augmentation arguments
  ROTATION_AUGMENTATION_BOUND = ((0, 0), (-np.pi, np.pi), (0, 0))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.1, 0.1), (0, 0), (-0.1, 0.1))

  ROTATION_AXIS = 'y'
  LOCFEAT_IDX = 1
  NUM_LABELS = 16  # Automatically subtract ignore labels after processed
  IGNORE_LABELS = (0, 1, 13, 14)  # void, sky, reserved, reserved

  # Split used in the Minkowski ConvNet, CVPR'19
  DATA_PATH_FILE = {
      DatasetPhase.Train: 'train_cvpr19.txt',
      DatasetPhase.Val: 'val_cvpr19.txt',
      DatasetPhase.Test: 'test_cvpr19.txt'
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
    if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
      self.CLIP_BOUND = self.TEST_CLIP_BOUND
    data_root = config.data.synthia_path
    data_paths = read_txt(osp.join('/checkpoint/jihou/data/synthia4d/splits', self.DATA_PATH_FILE[phase]))
    if phase == DatasetPhase.Train:
      data_paths = data_paths[:int(len(data_paths)*config.data.data_ratio)]
    data_paths = [d.split()[0] for d in data_paths]
    logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
    super().__init__(
        data_paths,
        data_root=data_root,
        input_transform=input_transform,
        target_transform=target_transform,
        ignore_label=config.data.ignore_label,
        return_transformation=config.data.return_transformation,
        augment_data=augment_data,
        elastic_distortion=elastic_distortion,
        config=config)

  def load_data(self, index):
    filepath = self.data_root / self.data_paths[index]
    plydata = PlyData.read(filepath)
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    feats = np.array([data['r'], data['g'], data['b']], dtype=np.float32).T
    labels = np.array(data['l'], dtype=np.int32)
    instances = np.zeros_like(labels)
    return coords, feats, labels, instances


class SynthiaCVPR15cmVoxelizationDataset(SynthiaVoxelizationDataset):
  pass


class SynthiaCVPR30cmVoxelizationDataset(SynthiaVoxelizationDataset):
  VOXEL_SIZE = 30


class SynthiaAllSequencesVoxelizationDataset(SynthiaVoxelizationDataset):
  DATA_PATH_FILE = {
      DatasetPhase.Train: 'train_raw.txt',
      DatasetPhase.Val: 'val_raw.txt',
      DatasetPhase.Test: 'test_raw.txt'
  }


class TestSynthia(unittest.TestCase):

  @debug_on()
  def test(self):
    from torch.utils.data import DataLoader
    from lib.utils import Timer
    from config import get_config
    config = get_config()

    dataset = SynthiaVoxelizationDataset(config)
    timer = Timer()

    data_loader = DataLoader(
        dataset=dataset,
        collate_fn=cfl_collate_fn_factory(limit_numpoints=False),
        num_workers=0,
        batch_size=4,
        shuffle=True)

    # Start from index 1
    # for i, batch in enumerate(data_loader, 1):
    iter = data_loader.__iter__()
    for i in range(100):
      timer.tic()
      batch = iter.next()
      print(batch, timer.toc())


if __name__ == '__main__':
  unittest.main()
