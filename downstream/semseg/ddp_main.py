# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import hydra
import torch

from lib.ddp_trainer import SegmentationTrainer
from lib.distributed import multi_proc_run

def single_proc_run(config):
  if not torch.cuda.is_available():
    raise Exception('No GPUs FOUND.')
  torch.manual_seed(config.misc.seed)
  torch.cuda.manual_seed(config.misc.seed)
  trainer = SegmentationTrainer(config)
  if config.train.is_train:
    trainer.train()
  else:
    trainer.test()

@hydra.main(config_path='config', config_name='default.yaml')
def main(config):
  # Convert to dict
  if config.misc.num_gpus > 1:
      multi_proc_run(config.misc.num_gpus, fun=single_proc_run, fun_args=(config,))
  else:
      single_proc_run(config)
   
if __name__ == '__main__':
  __spec__ = None
  os.environ['MKL_THREADING_LAYER'] = 'GNU'
  main()
