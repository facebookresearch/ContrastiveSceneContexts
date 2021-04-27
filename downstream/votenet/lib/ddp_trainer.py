import torch
import os
import sys
import logging
import numpy as np
import importlib
import warnings
import argparse

import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from models.loss_helper import get_loss as criterion
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
warnings.simplefilter(action='ignore', category=FutureWarning)
from models.backbone.pointnet2.pytorch_utils import BNMomentumScheduler
from models.dump_helper import dump_results, dump_results_
from models.ap_helper import APCalculator, parse_predictions, parse_groundtruths

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.serialization import default_restore_location
from lib.distributed import multi_proc_run, is_master_proc, get_world_size

class DetectionTrainer():
    def __init__(self, config):
        self.is_master = is_master_proc(get_world_size()) if get_world_size() > 1 else True
        self.cur_device = torch.cuda.current_device()

        # load the configurations
        self.setup_logging()
        if os.path.exists('config.yaml'):
            logging.info('===> Loading exsiting config file')
            config = OmegaConf.load('config.yaml')
            logging.info('===> Loaded exsiting config file')
        logging.info('===> Configurations')
        logging.info(config.pretty())

        # Create Dataset and Dataloader
        if config.data.dataset == 'sunrgbd':
            from datasets.sunrgbd.sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, MAX_NUM_OBJ
            from datasets.sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig
            dataset_config = SunrgbdDatasetConfig()
            train_dataset = SunrgbdDetectionVotesDataset('train', 
                num_points=config.data.num_points,
                augment=True,
                use_color=config.data.use_color, 
                use_height=(not config.data.no_height),
                use_v1=(not config.data.use_sunrgbd_v2))
            test_dataset = SunrgbdDetectionVotesDataset(config.test.phase, 
                num_points=config.data.num_points,
                augment=False,
                use_color=config.data.use_color, 
                use_height=(not config.data.no_height),
                use_v1=(not config.data.use_sunrgbd_v2))
        elif config.data.dataset == 'scannet':
            from datasets.scannet.scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
            from datasets.scannet.model_util_scannet import ScannetDatasetConfig
            dataset_config = ScannetDatasetConfig()
            train_dataset = ScannetDetectionDataset('train', 
                num_points=config.data.num_points,
                augment=True,
                use_color=config.data.use_color, 
                use_height=(not config.data.no_height),
                by_scenes=config.data.by_scenes,
                by_points=config.data.by_points)

            test_dataset = ScannetDetectionDataset(config.test.phase, 
                num_points=config.data.num_points,
                augment=False,
                use_color=config.data.use_color, 
                use_height=(not config.data.no_height))
        else:
            logging.info('Unknown dataset %s. Exiting...'%(config.data.dataset))
            exit(-1)

        COLLATE_FN = None
        if config.data.voxelization:
            from models.backbone.sparseconv.voxelized_dataset import VoxelizationDataset, collate_fn
            train_dataset = VoxelizationDataset(train_dataset, config.data.voxel_size)
            test_dataset = VoxelizationDataset(test_dataset, config.data.voxel_size)
            COLLATE_FN = collate_fn
        logging.info('training: {}, testing: {}'.format(len(train_dataset), len(test_dataset)))

        self.sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if get_world_size() > 1 else None
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.data.batch_size // config.misc.num_gpus,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=config.data.num_workers, 
            collate_fn=COLLATE_FN)

        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=1,
            shuffle=False, 
            num_workers=1, 
            collate_fn=COLLATE_FN)
        logging.info('train dataloader: {}, test dataloader: {}'.format(len(train_dataloader),len(test_dataloader)))

        # Init the model and optimzier
        MODEL = importlib.import_module('models.' + config.net.model) # import network module
        num_input_channel = int(config.data.use_color)*3 + int(not config.data.no_height)*1

        if config.net.model == 'boxnet':
            Detector = MODEL.BoxNet
        else:
            Detector = MODEL.VoteNet

        net = Detector(num_class=dataset_config.num_class,
                    num_heading_bin=dataset_config.num_heading_bin,
                    num_size_cluster=dataset_config.num_size_cluster,
                    mean_size_arr=dataset_config.mean_size_arr,
                    num_proposal=config.net.num_target,
                    input_feature_dim=num_input_channel,
                    vote_factor=config.net.vote_factor,
                    sampling=config.net.cluster_sampling,
                    backbone=config.net.backbone)

        if config.net.weights != '':
            #assert config.net.backbone == "sparseconv", "only support sparseconv"
            print('===> Loading weights: ' + config.net.weights)
            state = torch.load(config.net.weights, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            model = net
            if config.net.is_train:
                model = net.backbone_net
                if config.net.backbone == "sparseconv":
                    model = net.backbone_net.net
                    
            matched_weights = DetectionTrainer.load_state_with_same_shape(model, state['state_dict'])
            model_dict = model.state_dict()
            model_dict.update(matched_weights)
            model.load_state_dict(model_dict)

        net.to(self.cur_device)
        if get_world_size() > 1:
            net = torch.nn.parallel.DistributedDataParallel(
            module=net, device_ids=[self.cur_device], output_device=self.cur_device, broadcast_buffers=False) 

        # Load the Adam optimizer
        self.optimizer = optim.Adam(net.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
        # writer
        if self.is_master:
            self.writer = SummaryWriter(log_dir='tensorboard')
        self.config = config
        self.dataset_config = dataset_config
        self.net = net
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.best_mAP = -1

        # Used for AP calculation
        self.CONFIG_DICT = {'remove_empty_box':False, 'use_3d_nms':True,
                            'nms_iou':0.25, 'use_old_type_nms':False, 'cls_nms':True,
                            'per_class_proposal': True, 'conf_thresh':0.05, 'dataset_config': dataset_config}

        # Used for AP calculation
        self.CONFIG_DICT_TEST = {'remove_empty_box': (not config.test.faster_eval), 
                                 'use_3d_nms': config.test.use_3d_nms, 
                                 'nms_iou': config.test.nms_iou,
                                 'use_old_type_nms': config.test.use_old_type_nms, 
                                 'cls_nms': config.test.use_cls_nms, 
                                 'per_class_proposal': config.test.per_class_proposal,
                                 'conf_thresh': config.test.conf_thresh,
                                 'dataset_config': dataset_config}

        # Load checkpoint if there is any
        self.start_epoch = 0
        CHECKPOINT_PATH = os.path.join('checkpoint.tar')
        if os.path.isfile(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH)
            if get_world_size() > 1:
                _model = self.net.module
            else:
                _model = self.net
            _model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.best_mAP = checkpoint['best_mAP']
            logging.info("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, self.start_epoch))

        # Decay Batchnorm momentum from 0.5 to 0.999
        # note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
        BN_MOMENTUM_INIT = 0.5
        BN_MOMENTUM_MAX = 0.001
        BN_DECAY_STEP = config.optimizer.bn_decay_step
        BN_DECAY_RATE = config.optimizer.bn_decay_rate
        bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
        self.bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=self.start_epoch-1)

    def setup_logging(self):
        ch = logging.StreamHandler(sys.stdout)
        logging.getLogger().setLevel(logging.WARN)
        if self.is_master:
            logging.getLogger().setLevel(logging.INFO)
        logging.basicConfig(
            format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
            datefmt='%m/%d %H:%M:%S',
            handlers=[ch])

    @staticmethod
    def load_state_with_same_shape(model, weights):
        model_state = model.state_dict()
        if list(weights.keys())[0].startswith('module.'):
            print("Loading multigpu weights with module. prefix...")
            weights = {k.partition('module.')[2]:weights[k] for k in weights.keys()}

        if list(weights.keys())[0].startswith('encoder.'):
            logging.info("Loading multigpu weights with encoder. prefix...")
            weights = {k.partition('encoder.')[2]:weights[k] for k in weights.keys()}

        # print(weights.items())
        filtered_weights = {
            k: v for k, v in weights.items() if k in model_state  and v.size() == model_state[k].size()
        }
        print("Loading weights:" + ', '.join(filtered_weights.keys()))
        return filtered_weights
    
    @staticmethod
    def get_current_lr(epoch, config):
        lr = config.optimizer.learning_rate
        for i,lr_decay_epoch in enumerate(config.optimizer.lr_decay_steps):
            if epoch >= lr_decay_epoch:
                lr *= config.optimizer.lr_decay_rates[i]
        return lr

    @staticmethod
    def adjust_learning_rate(optimizer, epoch, config):
        lr = DetectionTrainer.get_current_lr(epoch, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train_one_epoch(self, epoch_cnt):
        stat_dict = {} # collect statistics
        DetectionTrainer.adjust_learning_rate(self.optimizer, epoch_cnt, self.config)
        self.bnm_scheduler.step() # decay BN momentum
        self.net.train() # set model to training mode
        for batch_idx, batch_data_label in enumerate(self.train_dataloader):
            for key in batch_data_label:
                if key == 'scan_name':
                    continue
                batch_data_label[key] = batch_data_label[key].cuda()

            # Forward pass
            self.optimizer.zero_grad()
            inputs = {'point_clouds': batch_data_label['point_clouds']}
            if 'voxel_coords' in batch_data_label:
                inputs.update({
                    'voxel_coords': batch_data_label['voxel_coords'],
                    'voxel_inds':   batch_data_label['voxel_inds'],
                    'voxel_feats':  batch_data_label['voxel_feats']})

            end_points = self.net(inputs)
            
            # Compute loss and gradients, update parameters.
            for key in batch_data_label:
                assert(key not in end_points)
                end_points[key] = batch_data_label[key]
            loss, end_points = criterion(end_points, self.dataset_config)
            loss.backward()
            self.optimizer.step()

            # Accumulate statistics and print out
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'ratio' in key:
                    if key not in stat_dict: stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

            batch_interval = 10
            if ((batch_idx+1) % batch_interval == 0) and self.is_master:
                logging.info(' ---- batch: %03d ----' % (batch_idx+1))
                for key in stat_dict:
                    self.writer.add_scalar('training/{}'.format(key), stat_dict[key]/batch_interval, 
                                          (epoch_cnt*len(self.train_dataloader)+batch_idx)*self.config.data.batch_size)
                for key in sorted(stat_dict.keys()):
                    logging.info('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                    stat_dict[key] = 0

    def evaluate_one_epoch(self, epoch_cnt):
        np.random.seed(0)
        stat_dict = {} # collect statistics

        ap_calculator = APCalculator(ap_iou_thresh=self.config.test.ap_iou, class2type_map=self.dataset_config.class2type)
        self.net.eval() # set model to eval mode (for bn and dp)
        for batch_idx, batch_data_label in enumerate(self.test_dataloader):
            if batch_idx % 10 == 0:
                logging.info('Eval batch: %d'%(batch_idx))
            for key in batch_data_label:
                if key == 'scan_name':
                    continue
                batch_data_label[key] = batch_data_label[key].cuda()
            
            # Forward pass
            inputs = {'point_clouds': batch_data_label['point_clouds']}
            if 'voxel_coords' in batch_data_label:
                inputs.update({
                    'voxel_coords': batch_data_label['voxel_coords'],
                    'voxel_inds':   batch_data_label['voxel_inds'],
                    'voxel_feats':  batch_data_label['voxel_feats']})

            with torch.no_grad():
                end_points = self.net(inputs)

            # Compute loss
            for key in batch_data_label:
                assert(key not in end_points)
                end_points[key] = batch_data_label[key]
            loss, end_points = criterion(end_points, self.dataset_config)

            # Accumulate statistics and print out
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'ratio' in key:
                    if key not in stat_dict: stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

            batch_pred_map_cls = parse_predictions(end_points, self.CONFIG_DICT) 
            batch_gt_map_cls = parse_groundtruths(end_points, self.CONFIG_DICT) 
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

            # Dump evaluation results for visualization
            if self.config.data.dump_results and batch_idx == 0 and epoch_cnt %10 == 0 and self.is_master:
                dump_results(end_points, 'results', self.dataset_config) 

        # Log statistics
        logging.info('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))
        if self.is_master:
            for key in sorted(stat_dict.keys()):
                self.writer.add_scalar('validation/{}'.format(key), stat_dict[key]/float(batch_idx+1),
                                (epoch_cnt+1)*len(self.train_dataloader)*self.config.data.batch_size)

        # Evaluate average precision
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            logging.info('eval %s: %f'%(key, metrics_dict[key]))
        if self.is_master:
            self.writer.add_scalar('validation/mAP{}'.format(self.config.test.ap_iou), metrics_dict['mAP'], (epoch_cnt+1)*len(self.train_dataloader)*self.config.data.batch_size)
        #mean_loss = stat_dict['loss']/float(batch_idx+1)

        return metrics_dict['mAP']

    def train(self):
        for epoch in range(self.start_epoch, self.config.optimizer.max_epoch):
            logging.info('**** EPOCH %03d ****' % (epoch))
            logging.info('Current learning rate: %f'%(DetectionTrainer.get_current_lr(epoch, self.config)))
            logging.info('Current BN decay momentum: %f'%(self.bnm_scheduler.lmbd(self.bnm_scheduler.last_epoch)))
            logging.info(str(datetime.now()))
            # Reset numpy seed.
            # REF: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed()
            if get_world_size() > 1:
                self.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)

            if epoch % 5 == 4 and self.is_master: # Eval every 5 epochs
                best_mAP = self.evaluate_one_epoch(epoch)

                if best_mAP > self.best_mAP:
                    self.best_mAP = best_mAP
                    # Save checkpoint
                    save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'best_mAP': self.best_mAP}

                    if get_world_size() > 1:
                        save_dict['state_dict'] = self.net.module.state_dict()
                    else:
                        save_dict['state_dict'] = self.net.state_dict()

                    torch.save(save_dict, 'checkpoint.tar')
                    OmegaConf.save(self.config, 'config.yaml')


    @staticmethod
    def write_to_benchmark(data, scene_name):
        from models.ap_helper import flip_axis_back_camera
        OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
        os.makedirs('benchmark_output', exist_ok=True)
        bsize = len(scene_name)
        for bsize_ in range(bsize):
            write_list = []
            cur_data = data[bsize_]
            cur_name = scene_name[bsize_]
            for class_id, bbox, score in cur_data:
                bbox = flip_axis_back_camera(bbox)
                minx = np.min(bbox[:,0])
                miny = np.min(bbox[:,1])
                minz = np.min(bbox[:,2])
                maxx = np.max(bbox[:,0])
                maxy = np.max(bbox[:,1])
                maxz = np.max(bbox[:,2])
                write_list.append([minx, miny, minz, maxx,maxy, maxz, OBJ_CLASS_IDS[class_id], score])

            np.savetxt(os.path.join('benchmark_output', cur_name+'.txt'), np.array(write_list))


    def test(self):
        if self.config.test.use_cls_nms:
            assert(self.config.test.use_3d_nms)

        AP_IOU_THRESHOLDS = self.config.test.ap_iou_thresholds
        logging.info(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed(0)
        stat_dict = {}
        ap_calculator_list = [APCalculator(iou_thresh, self.dataset_config.class2type) for iou_thresh in AP_IOU_THRESHOLDS]
        self.net.eval() # set model to eval mode (for bn and dp)
        for batch_idx, batch_data_label in enumerate(self.test_dataloader):
            if batch_idx % 10 == 0:
                print('Eval batch: %d'%(batch_idx))
            for key in batch_data_label:
                if key == 'scan_name':
                    continue
                batch_data_label[key] = batch_data_label[key].cuda()
            # Forward pass
            inputs = {'point_clouds': batch_data_label['point_clouds']}
            if 'voxel_coords' in batch_data_label:
                inputs.update({
                    'voxel_coords': batch_data_label['voxel_coords'],
                    'voxel_inds':   batch_data_label['voxel_inds'],
                    'voxel_feats':  batch_data_label['voxel_feats']})
            with torch.no_grad():
                end_points = self.net(inputs)

            # Compute loss
            for key in batch_data_label:
                assert(key not in end_points)
                end_points[key] = batch_data_label[key]
            loss, end_points = criterion(end_points, self.dataset_config)

            # Accumulate statistics and print out
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'ratio' in key:
                    if key not in stat_dict: stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

            batch_pred_map_cls = parse_predictions(end_points, self.CONFIG_DICT_TEST) 
            batch_gt_map_cls = parse_groundtruths(end_points, self.CONFIG_DICT_TEST) 
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

            # debug
            if self.config.test.write_to_benchmark:
                #from lib.utils.io3d import write_triangle_mesh
                #write_triangle_mesh(batch_data_label['point_clouds'][0].cpu().numpy(), None, None, batch_data_label['scan_name'][0]+'.ply')
                DetectionTrainer.write_to_benchmark(batch_pred_map_cls, batch_data_label['scan_name'])
            
            if self.config.test.save_vis:
                dump_results_(end_points, 'visualization', self.dataset_config)

        # Log statistics
        for key in sorted(stat_dict.keys()):
            logging.info('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

        # Evaluate average precision
        if not self.config.test.write_to_benchmark:
            for i, ap_calculator in enumerate(ap_calculator_list):
                logging.info('-'*10 + 'iou_thresh: %f'%(AP_IOU_THRESHOLDS[i]) + '-'*10)
                metrics_dict = ap_calculator.compute_metrics()
                for key in metrics_dict:
                    logging.info('eval %s: %f'%(key, metrics_dict[key]))

        mean_loss = stat_dict['loss']/float(batch_idx+1)
        return mean_loss
