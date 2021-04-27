import logging
import os
import shutil
import tempfile
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from lib.utils import Timer, AverageMeter, precision_at_one, fast_hist, per_class_iu, \
    get_prediction, get_torch_device, visualize_results, \
    permute_pointcloud, save_rotation_pred

from MinkowskiEngine import SparseTensor


from lib.bfs.bfs import Clustering
from datasets.evaluation.evaluate_semantic_instance import Evaluator as InstanceEvaluator
from datasets.evaluation.evaluate_semantic_label import Evaluator as SemanticEvaluator

def print_info(iteration,
               max_iteration,
               data_time,
               iter_time,
               losses=None,
               scores=None,
               ious=None,
               hist=None,
               ap_class=None,
               class_names=None):
  debug_str = "{}/{}: ".format(iteration + 1, max_iteration)
  debug_str += "Data time: {:.4f}, Iter time: {:.4f}".format(data_time, iter_time)

  acc = hist.diagonal() / hist.sum(1) * 100
  debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f})\t" \
      "Score {top1.val:.3f} (AVG: {top1.avg:.3f})\t" \
      "mIOU {mIOU:.3f} mAP {mAP:.3f} mAcc {mAcc:.3f}\n".format(
          loss=losses, top1=scores, mIOU=np.nanmean(ious),
          mAP=np.nanmean(ap_class), mAcc=np.nanmean(acc))
  if class_names is not None:
    debug_str += "\nClasses: " + " ".join(class_names) + '\n'
  debug_str += 'IOU: ' + ' '.join('{:.03f}'.format(i) for i in ious) + '\n'
  debug_str += 'mAP: ' + ' '.join('{:.03f}'.format(i) for i in ap_class) + '\n'
  debug_str += 'mAcc: ' + ' '.join('{:.03f}'.format(i) for i in acc) + '\n'

  logging.info(debug_str)


def average_precision(prob_np, target_np):
  num_class = prob_np.shape[1]
  label = label_binarize(target_np, classes=list(range(num_class)))
  with np.errstate(divide='ignore', invalid='ignore'):
    return average_precision_score(label, prob_np, average=None)


def nms(instances, instances_):
  instances_return = {}
  counter = 0
  for key in instances:
    label = instances[key]['label_id'].item()
    if label in [10, 12, 16]:
      continue
    instances_return[counter] = instances[key]
    counter += 1

  # dual set clustering, for some classes, w/o voting loss is better
  for key_ in instances_:
    label_ = instances_[key_]['label_id'].item()
    if label_ in [10, 12, 16]:
      instances_return[counter] = instances_[key_]
      counter += 1

  return instances_return


def test(model, data_loader, config):
  device = get_torch_device(config.misc.is_cuda)
  dataset = data_loader.dataset
  num_labels = dataset.NUM_LABELS
  global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()
  criterion = nn.CrossEntropyLoss(ignore_index=config.data.ignore_label)
  losses, scores, ious = AverageMeter(), AverageMeter(), 0
  aps = np.zeros((0, num_labels))
  hist = np.zeros((num_labels, num_labels))

  logging.info('===> Start testing')

  global_timer.tic()
  data_iter = data_loader.__iter__()
  max_iter = len(data_loader)
  max_iter_unique = max_iter



  ######################################################################################
  #  Added for Instance Segmentation
  ######################################################################################
  VALID_CLASS_IDS = torch.FloatTensor(dataset.VALID_CLASS_IDS).long()
  CLASS_LABELS_INSTANCE = dataset.CLASS_LABELS if config.misc.train_stuff else dataset.CLASS_LABELS_INSTANCE
  VALID_CLASS_IDS_INSTANCE = dataset.VALID_CLASS_IDS if config.misc.train_stuff else dataset.VALID_CLASS_IDS_INSTANCE
  IGNORE_LABELS_INSTANCE = dataset.IGNORE_LABELS if config.misc.train_stuff else dataset.IGNORE_LABELS_INSTANCE
  evaluator = InstanceEvaluator(CLASS_LABELS_INSTANCE, VALID_CLASS_IDS_INSTANCE)

  cluster_thresh = 1.5
  propose_points = 100
  score_func = torch.mean
  if config.test.evaluate_benchmark:
    cluster_thresh = 0.02
    propose_points = 250
    score_func = torch.median

  cluster = Clustering(ignored_labels=IGNORE_LABELS_INSTANCE, 
                        class_mapping=VALID_CLASS_IDS,
                        thresh=cluster_thresh, 
                        score_func=score_func,
                        propose_points=propose_points,
                        closed_points=300, 
                        min_points=50)
  if config.test.dual_set_cluster :
    # dual set clustering when submit to benchmark
    cluster_ = Clustering(ignored_labels=IGNORE_LABELS_INSTANCE, 
                        class_mapping=VALID_CLASS_IDS,
                        thresh=0.05, 
                        score_func=torch.mean,
                        propose_points=250,
                        closed_points=300, 
                        min_points=50)



  ######################################################################################


  # Fix batch normalization running mean and std
  model.eval()
  # Clear cache (when run in val mode, cleanup training cache)
  torch.cuda.empty_cache()
  with torch.no_grad():
    for iteration in range(max_iter):
      data_timer.tic()
      if config.data.return_transformation:
        coords, input, target, instances, transformation = data_iter.next()
      else:
        coords, input, target, instances = data_iter.next()
        transformation = None
      data_time = data_timer.toc(False)

      # Preprocess input
      iter_timer.tic()

      if config.net.wrapper_type != None:
        color = input[:, :3].int()
      if config.augmentation.normalize_color:
        input[:, :3] = input[:, :3] / 255. - 0.5
      sinput = SparseTensor(input, coords).to(device)

      # Feed forward
      inputs = (sinput,) if config.net.wrapper_type == None else (sinput, coords, color)
      pt_offsets, soutput, out_feats = model(*inputs)
      output = soutput.F

      pred = get_prediction(dataset, output, target).int()
      iter_time = iter_timer.toc(False)

      #####################################################################################
      #  Added for Instance Segmentation
      ######################################################################################
      if config.test.evaluate_benchmark:
          # ---------------- point level -------------------
          # voting loss for dual set clustering, w/o using ScoreNet
          scene_id = dataset.get_output_id(iteration)
          inverse_mapping = dataset.get_original_pointcloud(coords, transformation, iteration)
          vertices = inverse_mapping[1] + pt_offsets.feats[inverse_mapping[0]].cpu().numpy()
          features = output[inverse_mapping[0]]
          instances = cluster.get_instances(vertices, features)
          if config.test.dual_set_cluster:
              instances_ = cluster_.get_instances(inverse_mapping[1], features)
              instances = nms(instances, instances_)
          evaluator.add_prediction(instances, scene_id)
          # comment out when evaluate on benchmark format
          # evaluator.add_gt_in_benchmark_format(scene_id) 
          evaluator.write_to_benchmark(scene_id=scene_id, pred_inst=instances)
      else:
          # --------------- voxel level------------------
          vertices = coords.cpu().numpy()[:,1:] + pt_offsets.F.cpu().numpy() / dataset.VOXEL_SIZE
          clusterred_result = cluster.get_instances(vertices, output.clone().cpu())
          instance_ids = instances[0]['ids'] 
          gt_labels = target.clone()
          gt_labels[instance_ids == -1] = IGNORE_LABELS_INSTANCE[0] #invalid instance id is -1, map 0,1,255 labels to 0
          gt_labels = VALID_CLASS_IDS[gt_labels.long()]
          evaluator.add_gt((gt_labels*1000 + instance_ids).numpy(), iteration) # map invalid to invalid label, which is ignored anyway
          evaluator.add_prediction(clusterred_result, iteration)
      ######################################################################################

      target_np = target.numpy()
      num_sample = target_np.shape[0]
      target = target.to(device)

      cross_ent = criterion(output, target.long())
      losses.update(float(cross_ent), num_sample)
      scores.update(precision_at_one(pred, target), num_sample)
      hist += fast_hist(pred.cpu().numpy().flatten(), target_np.flatten(), num_labels)
      ious = per_class_iu(hist) * 100

      prob = torch.nn.functional.softmax(output, dim=1)
      ap = average_precision(prob.cpu().detach().numpy(), target_np)
      aps = np.vstack((aps, ap))
      # Due to heavy bias in class, there exists class with no test label at all
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ap_class = np.nanmean(aps, 0) * 100.

      if iteration % config.test.test_stat_freq == 0 and iteration > 0:
        reordered_ious = dataset.reorder_result(ious)
        reordered_ap_class = dataset.reorder_result(ap_class)
        class_names = dataset.get_classnames()
        print_info(
            iteration,
            max_iter_unique,
            data_time,
            iter_time,
            losses,
            scores,
            reordered_ious,
            hist,
            reordered_ap_class,
            class_names=class_names)

      if iteration % config.train.empty_cache_freq == 0:
        # Clear cache
        torch.cuda.empty_cache()

  global_time = global_timer.toc(False)

  reordered_ious = dataset.reorder_result(ious)
  reordered_ap_class = dataset.reorder_result(ap_class)
  class_names = dataset.get_classnames()
  print_info(
      iteration,
      max_iter_unique,
      data_time,
      iter_time,
      losses,
      scores,
      reordered_ious,
      hist,
      reordered_ap_class,
      class_names=class_names)


  logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

  mAP50 = 0.0
  #if not config.test.evaluate_benchmark:
  _, mAP50, _ = evaluator.evaluate()


  return losses.avg, scores.avg, np.nanmean(ap_class), np.nanmean(per_class_iu(hist)) * 100, mAP50
