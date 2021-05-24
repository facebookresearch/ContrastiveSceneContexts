# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import logging
import os
import sys
import torch
import logging
import torch.nn.functional as F

from torch import nn
from torch.serialization import default_restore_location
from tensorboardX import SummaryWriter
from MinkowskiEngine import SparseTensor
from omegaconf import OmegaConf

from lib.distributed import get_world_size, all_gather, is_master_proc
from models import load_model
from lib.test import test as test_
from lib.solvers import initialize_optimizer, initialize_scheduler
from datasets import load_dataset
from datasets.dataset import initialize_data_loader
from lib.utils import checkpoint, precision_at_one, Timer, AverageMeter, get_prediction, load_state_with_same_shape, count_parameters


class SegmentationTrainer:
    def __init__(self, config):

        self.is_master = is_master_proc(config.misc.num_gpus) if config.misc.num_gpus > 1 else True
        self.cur_device = torch.cuda.current_device()

        # load the configurations
        self.setup_logging()
        if os.path.exists('config.yaml'):
            logging.info('===> Loading exsiting config file')
            config = OmegaConf.load('config.yaml')
            logging.info('===> Loaded exsiting config file')
        logging.info('===> Configurations')
        logging.info(config.pretty())

        # dataloader
        DatasetClass = load_dataset(config.data.dataset)
        logging.info('===> Initializing dataloader')
        self.train_data_loader = initialize_data_loader(
            DatasetClass, config, phase=config.train.train_phase,
            num_workers=config.data.num_workers, augment_data=True,
            shuffle=True, repeat=True, batch_size=config.data.batch_size // config.misc.num_gpus,
            limit_numpoints=config.data.train_limit_numpoints)

        self.val_data_loader = initialize_data_loader(
            DatasetClass, config, phase=config.train.val_phase,
            num_workers=1, augment_data=False,
            shuffle=True, repeat=False,
            batch_size=1, limit_numpoints=False)
            
        self.test_data_loader = initialize_data_loader(
            DatasetClass, config, phase=config.test.test_phase,
            num_workers=config.data.num_workers, augment_data=False,
            shuffle=False, repeat=False,
            batch_size=config.data.test_batch_size, limit_numpoints=False)

        # Model initialization
        logging.info('===> Building model')
        num_in_channel = self.train_data_loader.dataset.NUM_IN_CHANNEL
        num_labels = self.train_data_loader.dataset.NUM_LABELS
        NetClass = load_model(config.net.model)
        model = NetClass(num_in_channel, num_labels, config)
        logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__, count_parameters(model)))
        logging.info(model)

        # Load weights if specified by the parameter.
        if config.net.weights != '':
            logging.info('===> Loading weights: ' + config.net.weights)
            state = torch.load(config.net.weights, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            matched_weights = load_state_with_same_shape(model, state['state_dict'])
            model_dict = model.state_dict()
            model_dict.update(matched_weights)
            model.load_state_dict(model_dict)

        model = model.cuda()
        if config.misc.num_gpus > 1:
            model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[self.cur_device], 
            output_device=self.cur_device,
            broadcast_buffers=False
            ) 

        self.config = config
        self.model = model
        if self.is_master:
            self.writer = SummaryWriter(log_dir='tensorboard')
        self.optimizer = initialize_optimizer(model.parameters(), config.optimizer)
        self.scheduler = initialize_scheduler(self.optimizer, config.optimizer)
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.data.ignore_label)

        checkpoint_fn = 'weights/weights.pth'
        self.best_val_miou, self.best_val_miou_iter = -1,1
        self.best_val_mAP, self.best_val_mAP_iter = -1,1
        self.curr_iter, self.epoch, self.is_training = 1, 1, True
        if os.path.isfile(checkpoint_fn):
            logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
            state = torch.load(checkpoint_fn, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            self.load_state(state['state_dict'])

            self.curr_iter = state['iteration'] + 1
            self.epoch = state['epoch']
            self.scheduler = initialize_scheduler(self.optimizer, config.optimizer, last_step=self.curr_iter)
            self.optimizer.load_state_dict(state['optimizer'])

            if 'best_val_miou' in state:
              self.best_val_miou = state['best_val_miou']
            if 'best_val_mAP' in state:
              self.best_val_mAP = state['best_val_mAP']

            logging.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_fn, state['epoch']))
        else:
            logging.info("=> no weights.pth")

    def setup_logging(self):
        ch = logging.StreamHandler(sys.stdout)
        logging.getLogger().setLevel(logging.WARN)
        if self.is_master:
            logging.getLogger().setLevel(logging.INFO)
        logging.basicConfig(
            format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
            datefmt='%m/%d %H:%M:%S',
            handlers=[ch])

    def load_state(self, state):
        if get_world_size() > 1:
            _model = self.model.module
        else:
            _model = self.model  
        _model.load_state_dict(state)

    def set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.config.misc.seed + self.curr_iter
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def test(self):
        return test_(self.model, self.test_data_loader, self.config)
    
    def validate(self):
        val_loss, val_score, _, val_miou, val_mAP = test_(self.model, self.val_data_loader, self.config)
        self.writer.add_scalar('val/miou', val_miou, self.curr_iter)
        self.writer.add_scalar('val/loss', val_loss, self.curr_iter)
        self.writer.add_scalar('val/precision_at_1', val_score, self.curr_iter)
        self.writer.add_scalar('val/mAP@0.5', val_mAP, self.curr_iter)

        if val_miou > self.best_val_miou:
            self.best_val_miou = val_miou
            self.best_val_iou_iter = self.curr_iter
            checkpoint(self.model, self.optimizer, self.epoch, self.curr_iter, self.config, 
                        self.best_val_miou, self.best_val_mAP, "miou")
            logging.info("Current best mIoU: {:.3f} at iter {}".format(self.best_val_miou, self.best_val_miou_iter))

        if val_mAP > self.best_val_mAP:
            self.best_val_mAP = val_mAP
            self.best_val_mAP_iter = self.curr_iter
            checkpoint(self.model, self.optimizer, self.epoch, self.curr_iter, self.config, 
                        self.best_val_miou, self.best_val_mAP, "mAP")
            logging.info("Current best mAP@0.5: {:.3f} at iter {}".format(self.best_val_mAP, self.best_val_mAP_iter))

        checkpoint(self.model, self.optimizer, self.epoch, self.curr_iter, self.config, 
                   self.best_val_miou, self.best_val_mAP)

    def train(self):
        # Set up the train flag for batch normalization
        self.model.train()

        # Configuration
        data_timer, iter_timer = Timer(), Timer()
        fw_timer, bw_timer, ddp_timer = Timer(), Timer(), Timer()
        data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
        fw_time_avg, bw_time_avg, ddp_time_avg = AverageMeter(), AverageMeter(), AverageMeter()

        scores = AverageMeter()
        losses = {
            'semantic_loss': AverageMeter(),
            'offset_dir_loss': AverageMeter(),
            'offset_norm_loss': AverageMeter(),
            'total_loss': AverageMeter()
        }

        # Train the network
        logging.info('===> Start training on {} GPUs, batch-size={}'.format(
            get_world_size(), self.config.data.batch_size))

        data_iter = self.train_data_loader.__iter__()  # (distributed) infinite sampler
        while self.is_training:
            for _ in range(len(self.train_data_loader) // self.config.optimizer.iter_size):
                self.optimizer.zero_grad()
                data_time, batch_score = 0, 0
                batch_losses = {
                'semantic_loss': 0.0,
                'offset_dir_loss': 0.0,
                'offset_norm_loss': 0.0,
                'total_loss': 0.0}
                iter_timer.tic()

                # set random seed for every iteration for trackability
                self.set_seed()

                for sub_iter in range(self.config.optimizer.iter_size):
                    # Get training data
                    data_timer.tic()
                    if self.config.data.return_transformation:
                        coords, input, target, instances, _ = data_iter.next()
                    else:
                        coords, input, target, instances = data_iter.next()

                    # Preprocess input
                    color = input[:, :3].int()
                    if self.config.augmentation.normalize_color:
                      input[:, :3] = input[:, :3] / 255. - 0.5
                    sinput = SparseTensor(input, coords).to(self.cur_device)

                    data_time += data_timer.toc(False)
                    # Feed forward
                    fw_timer.tic()

                    inputs = (sinput,) 
                    pt_offsets, soutput, _ = self.model(*inputs)
                    # The output of the network is not sorted
                    target = target.long().to(self.cur_device)
                    semantic_loss = self.criterion(soutput.F, target.long())
                    total_loss = semantic_loss

                    #-----------------offset loss----------------------
                    ## pt_offsets: (N, 3), float, cuda
                    ## coords: (N, 3), float32
                    ## centers: (N, 3), float32 tensor 
                    ## instance_ids: (N), long
                    centers = np.concatenate([instance['center'] for instance in instances])
                    instance_ids = np.concatenate([instance['ids'] for instance in instances])

                    centers = torch.from_numpy(centers).cuda()
                    instance_ids = torch.from_numpy(instance_ids).cuda().long()

                    gt_offsets = centers - coords[:,1:].cuda()   # (N, 3)
                    gt_offsets *= self.train_data_loader.dataset.VOXEL_SIZE
                    pt_diff = pt_offsets.F - gt_offsets   # (N, 3)
                    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
                    valid = (instance_ids != -1).float()
                    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

                    gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
                    gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
                    pt_offsets_norm = torch.norm(pt_offsets.F, p=2, dim=1)
                    pt_offsets_ = pt_offsets.F / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
                    direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
                    offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)
                    total_loss += offset_norm_loss + offset_dir_loss
    
                    # Compute and accumulate gradient
                    total_loss /= self.config.optimizer.iter_size

                    pred = get_prediction(self.train_data_loader.dataset, soutput.F, target)
                    score = precision_at_one(pred, target)

                    # bp the loss
                    fw_timer.toc(False)
                    bw_timer.tic()
                    total_loss.backward()
                    bw_timer.toc(False)

                    # gather information
                    logging_output = {'total_loss': total_loss.item(), 'semantic_loss': semantic_loss.item(), 'score': score / self.config.optimizer.iter_size}
                    logging_output['offset_dir_loss'] = offset_dir_loss.item()
                    logging_output['offset_norm_loss'] = offset_norm_loss.item()

                    ddp_timer.tic()
                    if self.config.misc.num_gpus > 1:
                      logging_output = all_gather(logging_output)
                      logging_output = {w: np.mean([
                            a[w] for a in logging_output]
                          ) for w in logging_output[0]}

                    batch_losses['total_loss'] += logging_output['total_loss']
                    batch_losses['semantic_loss'] += logging_output['semantic_loss']
                    batch_losses['offset_dir_loss'] += logging_output['offset_dir_loss']
                    batch_losses['offset_norm_loss'] += logging_output['offset_norm_loss']
                    batch_score += logging_output['score']

                    ddp_timer.toc(False)

                # Update number of steps
                self.optimizer.step()
                self.scheduler.step()

                data_time_avg.update(data_time)
                iter_time_avg.update(iter_timer.toc(False))
                fw_time_avg.update(fw_timer.diff)
                bw_time_avg.update(bw_timer.diff)
                ddp_time_avg.update(ddp_timer.diff)

                losses['total_loss'].update(batch_losses['total_loss'], target.size(0))
                losses['semantic_loss'].update(batch_losses['semantic_loss'], target.size(0))
                losses['offset_dir_loss'].update(batch_losses['offset_dir_loss'], target.size(0))
                losses['offset_norm_loss'].update(batch_losses['offset_norm_loss'], target.size(0))
                scores.update(batch_score, target.size(0))

                if self.curr_iter >= self.config.optimizer.max_iter:
                  self.is_training = False
                  break

                if self.curr_iter % self.config.train.stat_freq == 0 or self.curr_iter == 1:
                    lrs = ', '.join(['{:.3e}'.format(x) for x in self.scheduler.get_last_lr()])
                    debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}, Sem {:.4f}, dir {:.4f}, norm {:.4f}\tLR: {}\t".format(
                                self.epoch, self.curr_iter, len(self.train_data_loader) // self.config.optimizer.iter_size, 
                                losses['total_loss'].avg, losses['semantic_loss'].avg,
                                losses['offset_dir_loss'].avg, losses['offset_norm_loss'].avg, lrs)
                    debug_str += "Score {:.3f}\tData time: {:.4f}, Forward time: {:.4f}, Backward time: {:.4f}, DDP time: {:.4f}, Total iter time: {:.4f}".format(
                                  scores.avg, data_time_avg.avg, fw_time_avg.avg, bw_time_avg.avg, ddp_time_avg.avg, iter_time_avg.avg)
                    logging.info(debug_str)
                    # Reset timers
                    data_time_avg.reset()
                    iter_time_avg.reset()

                    # Write logs
                    if self.is_master:
                      self.writer.add_scalar('train/loss', losses['total_loss'].avg, self.curr_iter)
                      self.writer.add_scalar('train/semantic_loss', losses['semantic_loss'].avg, self.curr_iter)
                      self.writer.add_scalar('train/offset_dir_loss', losses['offset_dir_loss'].avg, self.curr_iter)
                      self.writer.add_scalar('train/offset_norm_loss', losses['offset_norm_loss'].avg, self.curr_iter)
                      self.writer.add_scalar('train/precision_at_1', scores.avg, self.curr_iter)
                      self.writer.add_scalar('train/learning_rate', self.scheduler.get_last_lr()[0], self.curr_iter)

                    # clear loss
                    losses['total_loss'].reset()
                    losses['semantic_loss'].reset()
                    losses['offset_dir_loss'].reset()
                    losses['offset_norm_loss'].reset()
                    scores.reset()

                # Validation
                if self.curr_iter % self.config.train.val_freq == 0 and self.is_master:
                    self.validate()
                    self.model.train()

                if self.curr_iter % self.config.train.empty_cache_freq == 0:
                  # Clear cache
                  torch.cuda.empty_cache()

                # End of iteration
                self.curr_iter += 1

            self.epoch += 1

        # Explicit memory cleanup
        if hasattr(data_iter, 'cleanup'):
            data_iter.cleanup()

        # Save the final model
        if self.is_master:
            self.validate()
