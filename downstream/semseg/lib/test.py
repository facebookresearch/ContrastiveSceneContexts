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
from datasets.evaluation.evaluate_semantic_label import Evaluator

from lib.utils import Timer, AverageMeter, precision_at_one, fast_hist, per_class_iu, \
    get_prediction, get_torch_device, visualize_results, \
    permute_pointcloud, save_rotation_pred

from MinkowskiEngine import SparseTensor

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



  #------------------------------- add -------------------------------------
  VALID_CLASS_IDS = torch.FloatTensor(dataset.VALID_CLASS_IDS).long()
  # Fix batch normalization running mean and std
  model.eval()

  # Clear cache (when run in val mode, cleanup training cache)
  torch.cuda.empty_cache()
  
  if config.test.save_features:
    save_feat_dir = config.test.save_feat_dir
    os.makedirs(save_feat_dir, exist_ok=True)

  with torch.no_grad():
    for iteration in range(max_iter):
      data_timer.tic()
      if config.data.return_transformation:
        coords, input, target, transformation = data_iter.next()
      else:
        coords, input, target = data_iter.next()
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
      soutput, out_feats = model(*inputs)
      output = soutput.F

      pred = get_prediction(dataset, output, target).int()


       
      if config.test.evaluate_benchmark:
          # ---------------- point level -------------------
          scene_id = dataset.get_output_id(iteration)
          inverse_mapping = dataset.get_original_pointcloud(coords, transformation, iteration)
          CLASS_MAP = np.array(dataset.VALID_CLASS_IDS)
          pred_points = CLASS_MAP[pred.cpu().numpy()][inverse_mapping[0]]
          # for benchmark
          Evaluator.write_to_benchmark(scene_id=scene_id, pred_ids=pred_points)

      iter_time = iter_timer.toc(False)

      if config.test.save_features:
        dataset.save_features(coords, out_feats.F, transformation, iteration, save_feat_dir)

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


  return losses.avg, scores.avg, np.nanmean(ap_class), np.nanmean(per_class_iu(hist)) * 100
