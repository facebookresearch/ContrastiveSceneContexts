# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import json
import logging
import torch
from omegaconf import OmegaConf

from easydict import EasyDict as edict

import lib.multiprocessing_utils as mpu
import hydra

from lib.ddp_trainer import PointNCELossTrainer, PartitionPointNCELossTrainer, PartitionPointNCELossTrainerPointNet

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")

def get_trainer(trainer):
  if trainer == 'PointNCELossTrainer':
    return PointNCELossTrainer
  elif trainer == 'PartitionPointNCELossTrainer':
    return PartitionPointNCELossTrainer
  elif trainer == 'PartitionPointNCELossTrainerPointNet':
    return PartitionPointNCELossTrainerPointNet
  else:
    raise ValueError(f'Trainer {trainer} not found')

@hydra.main(config_path='config', config_name='defaults.yaml')
def main(config):
  if os.path.exists('config.yaml'):
    logging.info('===> Loading exsiting config file')
    config = OmegaConf.load('config.yaml')
    logging.info('===> Loaded exsiting config file')
  logging.info('===> Configurations')
  logging.info(config.pretty())

  # Convert to dict
  if config.misc.num_gpus > 1:
      mpu.multi_proc_run(config.misc.num_gpus,
              fun=single_proc_run, fun_args=(config,))
  else:
      single_proc_run(config)

def single_proc_run(config):
  from lib.ddp_data_loaders import make_data_loader

  train_loader = make_data_loader(
      config,
      int(config.trainer.batch_size / config.misc.num_gpus),
      num_threads=int(config.misc.train_num_thread / config.misc.num_gpus))

  Trainer = get_trainer(config.trainer.trainer)
  trainer = Trainer(config=config, data_loader=train_loader)

  if config.misc.is_train:
    trainer.train()
  else:
    trainer.test()


if __name__ == "__main__":
  os.environ['MKL_THREADING_LAYER'] = 'GNU'
  main()
