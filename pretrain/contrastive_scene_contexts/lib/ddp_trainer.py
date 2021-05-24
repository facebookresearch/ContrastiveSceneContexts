# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import gc
import logging
import numpy as np
import json
from omegaconf import OmegaConf
import torch.nn as nn

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from lib.data_sampler import InfSampler, DistributedInfSampler

from model import load_model
from lib.timer import Timer, AverageMeter

import MinkowskiEngine as ME

import lib.distributed as du
import torch.distributed as dist

from lib.criterion import NCESoftmaxLoss

from torch.serialization import default_restore_location

torch.autograd.set_detect_anomaly(True)

LARGE_NUM = 1e9

def apply_transform(pts, trans):
  voxel_size = 0.025
  R = trans[:3, :3]
  T = trans[:3, 3]
  pts = pts * voxel_size
  pts = torch.matmul(pts - T, torch.inverse(R.T))
  pts = pts - torch.mean(pts, 0)
  pts = pts / voxel_size
  return pts

def _hash(arr, M):
  if isinstance(arr, np.ndarray):
    N, D = arr.shape
  else:
    N, D = len(arr[0]), len(arr)

  hash_vec = np.zeros(N, dtype=np.int64)
  for d in range(D):
    if isinstance(arr, np.ndarray):
      hash_vec += arr[:, d] * M**d
    else:
      hash_vec += arr[d] * M**d
  return hash_vec

def load_state(model, weights, lenient_weight_loading=False):
  if du.get_world_size() > 1:
      _model = model.module
  else:
      _model = model  

  if lenient_weight_loading:
    model_state = _model.state_dict()
    filtered_weights = {
        k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
    }
    logging.info("Load weights:" + ', '.join(filtered_weights.keys()))
    weights = model_state
    weights.update(filtered_weights)

  _model.load_state_dict(weights, strict=True)

def shuffle_loader(data_loader, cur_epoch):
  assert isinstance(data_loader.sampler, (RandomSampler, InfSampler, DistributedSampler, DistributedInfSampler))
  if isinstance(data_loader.sampler, DistributedSampler):
    data_loader.sampler.set_epoch(cur_epoch)

class ContrastiveLossTrainer:
  def __init__(
      self,
      config,
      data_loader):
    assert config.misc.use_gpu and torch.cuda.is_available(), "DDP mode must support GPU"
    num_feats = 3  # always 3 for finetuning.

    self.is_master = du.is_master_proc(config.misc.num_gpus) if config.misc.num_gpus > 1 else True

    # Model initialization
    self.cur_device = torch.cuda.current_device()
    Model = load_model(config.net.model)
    model = Model(
        num_feats,
        config.net.model_n_out,
        config,
        D=3)
    model = model.cuda(device=self.cur_device)
    if config.misc.num_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=[self.cur_device],
                output_device=self.cur_device,
                broadcast_buffers=False,
        )

    self.config = config
    self.model = model

    self.optimizer = getattr(optim, config.opt.optimizer)(
        model.parameters(),
        lr=config.opt.lr,
        momentum=config.opt.momentum,
        weight_decay=config.opt.weight_decay)

    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.opt.exp_gamma)
    self.curr_iter = 0
    self.batch_size = data_loader.batch_size
    self.data_loader = data_loader

    self.neg_thresh = config.trainer.neg_thresh
    self.pos_thresh = config.trainer.pos_thresh

    #---------------- optional: resume checkpoint by given path ----------------------
    if config.net.weight:
        if self.is_master:
          logging.info('===> Loading weights: ' + config.net.weight)
        state = torch.load(config.net.weight, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        load_state(model, state['state_dict'], config.misc.lenient_weight_loading)
        if self.is_master:
          logging.info('===> Loaded weights: ' + config.net.weight)

    #---------------- default: resume checkpoint in current folder ----------------------
    checkpoint_fn = 'weights/weights.pth'
    if osp.isfile(checkpoint_fn):
      if self.is_master:
        logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
      state = torch.load(checkpoint_fn, map_location=lambda s, l: default_restore_location(s, 'cpu'))
      self.curr_iter = state['curr_iter']
      load_state(model, state['state_dict'])
      self.optimizer.load_state_dict(state['optimizer'])
      self.scheduler.load_state_dict(state['scheduler'])
      if self.is_master:
        logging.info("=> loaded checkpoint '{}' (curr_iter {})".format(checkpoint_fn, state['curr_iter']))
    else:
      logging.info("=> no checkpoint found at '{}'".format(checkpoint_fn))

    if self.is_master:
        self.writer = SummaryWriter(logdir='logs')
        if not os.path.exists('weights'):
          os.makedirs('weights', mode=0o755)
        OmegaConf.save(config, 'config.yaml')

    # added
    from lib.shape_context import ShapeContext
    self.partitioner = ShapeContext(r1=config.shape_context.r1, 
                                    r2=config.shape_context.r2,
                                    nbins_xy=config.shape_context.nbins_xy,
                                    nbins_zy=config.shape_context.nbins_zy)


  def pdist(self, A, B):
    D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    return torch.sqrt(D2 + 1e-7)

  def _save_checkpoint(self, curr_iter, filename='checkpoint'):
    if not self.is_master:
        return
    _model = self.model.module if du.get_world_size() > 1 else self.model
    state = {
        'curr_iter': curr_iter,
        'state_dict': _model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
    }
    filepath = os.path.join('weights', f'{filename}.pth')
    logging.info("Saving checkpoint: {} ...".format(filepath))
    torch.save(state, filepath)
    # Delete symlink if it exists
    if os.path.exists('weights/weights.pth'):
      os.remove('weights/weights.pth')
    # Create symlink
    os.system('ln -s {}.pth weights/weights.pth'.format(filename))

class PointNCELossTrainer(ContrastiveLossTrainer):

  def __init__(
      self,
      config,
      data_loader):
    ContrastiveLossTrainer.__init__(self, config, data_loader)
    
    self.T = config.misc.nceT
    self.npos = config.misc.npos

    self.stat_freq = config.trainer.stat_freq
    self.lr_update_freq = config.trainer.lr_update_freq
    self.checkpoint_freq = config.trainer.checkpoint_freq
  
  def compute_loss(self, q, k, mask=None):
    npos = q.shape[0] 
    logits = torch.mm(q, k.transpose(1, 0)) # npos by npos
    labels = torch.arange(npos).cuda().long()
    out = torch.div(logits, self.T)
    out = out.squeeze().contiguous()
    if mask != None:
      out = out - LARGE_NUM * mask.float()
    criterion = NCESoftmaxLoss().cuda()
    loss = criterion(out, labels)
    return loss

  def train(self):

    curr_iter = self.curr_iter
    data_loader_iter = self.data_loader.__iter__()
    data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
    
    while (curr_iter < self.config.opt.max_iter):

      curr_iter += 1
      epoch = curr_iter / len(self.data_loader)

      batch_loss = self._train_iter(data_loader_iter, [data_meter, data_timer, total_timer])

      # update learning rate
      if curr_iter % self.lr_update_freq == 0 or curr_iter == 1:
        lr = self.scheduler.get_last_lr()
        self.scheduler.step()

      # Print logs
      if curr_iter % self.stat_freq == 0 and self.is_master:
        self.writer.add_scalar('train/loss', batch_loss['loss'], curr_iter)
        logging.info(
            "Train Epoch: {:.3f} [{}/{}], Current Loss: {:.3e}"
            .format(epoch, curr_iter,
                    len(self.data_loader), batch_loss['loss']) +
            "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}, LR: {}".format(
                data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg, self.scheduler.get_last_lr()))
        data_meter.reset()
        total_timer.reset()

      # save checkpoint
      if self.is_master and curr_iter % self.checkpoint_freq == 0:
        lr = self.scheduler.get_last_lr()
        logging.info(f" Epoch: {epoch}, LR: {lr}")
        checkpoint_name = 'checkpoint'
        if not self.config.trainer.overwrite_checkpoint:
          checkpoint_name += '_{}'.format(curr_iter)
        self._save_checkpoint(curr_iter, checkpoint_name)


  def _train_iter(self, data_loader_iter, timers):
    data_meter, data_timer, total_timer = timers
    
    self.optimizer.zero_grad()
    batch_loss = {
      'loss': 0.0, 
    }
    data_time = 0
    total_timer.tic()
    
    data_timer.tic()
    input_dict = data_loader_iter.next()
    data_time += data_timer.toc(average=False)

    sinput0 = ME.SparseTensor(
        input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.cur_device)
    F0 = self.model(sinput0)
    F0 = F0.F

    sinput1 = ME.SparseTensor(
        input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.cur_device)
    F1  = self.model(sinput1)
    F1 = F1.F

    N0, N1 = input_dict['pcd0'].shape[0], input_dict['pcd1'].shape[0]
    pos_pairs = input_dict['correspondences'].to(self.cur_device)
    
    q_unique, count = pos_pairs[:, 0].unique(return_counts=True)
    uniform = torch.distributions.Uniform(0, 1).sample([len(count)]).to(self.cur_device)
    off = torch.floor(uniform*count).long()
    cums = torch.cat([torch.tensor([0], device=self.cur_device), torch.cumsum(count, dim=0)[0:-1]], dim=0)
    k_sel = pos_pairs[:, 1][off+cums]

    if self.npos < q_unique.shape[0]:
        sampled_inds = np.random.choice(q_unique.shape[0], self.npos, replace=False)
        q_unique = q_unique[sampled_inds]
        k_sel = k_sel[sampled_inds]

    q = F0[q_unique.long()]
    k = F1[k_sel.long()]
    loss = self.compute_loss(q,k)

    loss.backward()

    result = {"loss": loss}
    if self.config.misc.num_gpus > 1:
      result = du.scaled_all_reduce_dict(result, self.config.misc.num_gpus)
    batch_loss['loss'] += result["loss"].item()

    self.optimizer.step()

    torch.cuda.empty_cache()
    total_timer.toc()
    data_meter.update(data_time)
    return batch_loss

class PartitionPointNCELossTrainer(PointNCELossTrainer):
  def _train_iter(self, data_loader_iter, timers):

    # optimizer and loss
    self.optimizer.zero_grad()
    batch_loss = {
      'loss': 0.0, 
    }
    loss = 0 

    # timing
    data_meter, data_timer, total_timer = timers
    data_time = 0
    total_timer.tic()
    data_timer.tic()
    input_dict = data_loader_iter.next()
    data_time += data_timer.toc(average=False)

    # network forwarding
    sinput0 = ME.SparseTensor(
        input_dict['sinput0_F'], coords=input_dict['sinput0_C']).to(self.cur_device)
    F0 = self.model(sinput0)
    F0 = F0.F

    sinput1 = ME.SparseTensor(
        input_dict['sinput1_F'], coords=input_dict['sinput1_C']).to(self.cur_device)
    F1 = self.model(sinput1)
    F1 = F1.F

    # get positive pairs
    pos_pairs = input_dict['correspondences'].to(self.cur_device)
    q_unique, count = pos_pairs[:, 0].unique(return_counts=True)
    uniform = torch.distributions.Uniform(0, 1).sample([len(count)]).to(self.cur_device)
    off = torch.floor(uniform*count).long()
    cums = torch.cat([torch.tensor([0], device=self.cur_device), torch.cumsum(count, dim=0)[0:-1]], dim=0)
    k_sel = pos_pairs[:, 1][off+cums]

    # iterate batch
    source_batch_ids = input_dict['sinput0_C'][q_unique.long()][:,0].float().cuda()
    for batch_id in range(self.batch_size):
      # batch mask
      mask = (source_batch_ids == batch_id)
      q_unique_batch = q_unique[mask]
      k_sel_batch = k_sel[mask]

      # sampling points in current scene
      if self.npos < q_unique_batch.shape[0]:
          sampled_inds = np.random.choice(q_unique_batch.shape[0], self.npos, replace=False)
          q_unique_batch = q_unique_batch[sampled_inds]
          k_sel_batch = k_sel_batch[sampled_inds]

      q = F0[q_unique_batch.long()]
      k = F1[k_sel_batch.long()]
      npos = q.shape[0] 
      if npos == 0:
        logging.info('partitionTrainer: no points in this batch')
        continue

      source_xyz = input_dict['sinput0_C'][q_unique_batch.long()][:,1:].float().cuda()
      
      if self.config.data.world_space:
        T0 = input_dict['T0'][batch_id].cuda()
        source_xyz = apply_transform(source_xyz, T0)
        
      if self.config.shape_context.fast_partition:
        source_partition = self.partitioner.compute_partitions_fast(source_xyz)
      else:
        source_partition = self.partitioner.compute_partitions(source_xyz)

      for partition_id in range(self.partitioner.partitions):
        factor = 1.0
        if self.config.shape_context.weight_inner and partition_id < int(self.partitioner.partitions/2):
          factor = 2.0
        mask_q = (source_partition == partition_id)
        mask_q.fill_diagonal_(True)
        loss += factor * self.compute_loss(q, k, ~mask_q) / (self.partitioner.partitions * self.batch_size)

    loss.backward()

    result = {"loss": loss}
    if self.config.misc.num_gpus > 1:
      result = du.scaled_all_reduce_dict(result, self.config.misc.num_gpus)
    batch_loss['loss'] += result["loss"].item()
    self.optimizer.step()

    torch.cuda.empty_cache()
    total_timer.toc()
    data_meter.update(data_time)
    return batch_loss


class PartitionPointNCELossTrainerPointNet(PointNCELossTrainer):
  def _train_iter(self, data_loader_iter, timers):

    # optimizer and loss
    self.optimizer.zero_grad()
    batch_loss = {
      'loss': 0.0, 
    }
    loss = 0 

    # timing
    data_meter, data_timer, total_timer = timers
    data_time = 0
    total_timer.tic()
    data_timer.tic()
    input_dict = data_loader_iter.next()
    data_time += data_timer.toc(average=False)

    # network forwarding
    points = input_dict['sinput0_C']
    feats = input_dict['sinput0_F']
    points0 = []
    for batch_id in points[:,0].unique():
      mask = points[:,0] == batch_id
      points0.append(points[mask, 1:])
    points0 = torch.stack(points0).cuda()
    F0 = self.model(points0)
    F0 = F0.transpose(1,2).contiguous()
    F0 = F0.view(-1, 32)

    points = input_dict['sinput1_C']
    feats = input_dict['sinput1_F']
    points1 = []
    for batch_id in points[:,0].unique():
      mask = points[:,0] == batch_id
      points1.append(points[mask, 1:])
    points1 = torch.stack(points1).cuda()
    F1 = self.model(points1)
    F1 = F1.transpose(1,2).contiguous()
    F1 = F1.view(-1, 32)

    # get positive pairs
    pos_pairs = input_dict['correspondences'].to(self.cur_device)
    q_unique, count = pos_pairs[:, 0].unique(return_counts=True)
    uniform = torch.distributions.Uniform(0, 1).sample([len(count)]).to(self.cur_device)
    off = torch.floor(uniform*count).long()
    cums = torch.cat([torch.tensor([0], device=self.cur_device), torch.cumsum(count, dim=0)[0:-1]], dim=0)
    k_sel = pos_pairs[:, 1][off+cums]

    # iterate batch
    source_batch_ids = input_dict['sinput0_C'][q_unique.long()][:,0].float().cuda()
    for batch_id in range(self.batch_size):
      # batch mask
      mask = (source_batch_ids == batch_id)
      q_unique_batch = q_unique[mask]
      k_sel_batch = k_sel[mask]

      # sampling points in current scene
      if self.npos < q_unique_batch.shape[0]:
          sampled_inds = np.random.choice(q_unique_batch.shape[0], self.npos, replace=False)
          q_unique_batch = q_unique_batch[sampled_inds]
          k_sel_batch = k_sel_batch[sampled_inds]

      q = F0[q_unique_batch.long()]
      k = F1[k_sel_batch.long()]
      npos = q.shape[0] 
      if npos == 0:
        logging.info('partitionTrainer: no points in this batch')
        continue
        
      source_xyz = input_dict['sinput0_C'][q_unique_batch.long()][:,1:].float().cuda()
      if self.config.data.world_space:
          T0 = input_dict['T0'][batch_id].cuda()
          source_xyz = apply_transform(source_xyz, T0)

      if self.config.shape_context.fast_partition:
        source_partition = self.partitioner.compute_partitions_fast(source_xyz)
      else:
        source_partition = self.partitioner.compute_partitions(source_xyz)

      for partition_id in range(self.partitioner.partitions):
        factor = 1.0
        if self.config.shape_context.weight_inner and partition_id < int(self.partitioner.partitions/2):
          factor = 2.0
        mask_q = (source_partition == partition_id)
        mask_q.fill_diagonal_(True)
        loss += factor * self.compute_loss(q, k, ~mask_q) / (self.partitioner.partitions * self.batch_size)

    loss.backward()

    result = {"loss": loss}
    if self.config.misc.num_gpus > 1:
      result = du.scaled_all_reduce_dict(result, self.config.misc.num_gpus)
    batch_loss['loss'] += result["loss"].item()
    self.optimizer.step()

    torch.cuda.empty_cache()
    total_timer.toc()
    data_meter.update(data_time)
    return batch_loss
