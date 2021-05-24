# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os, sys, argparse
import inspect
from copy import deepcopy
from evaluate_object_detection_helper import eval_det
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import util
import util_3d

parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', required=True, help='path to directory of predicted .txt files')
parser.add_argument('--gt_path', required=True, help='path to directory of gt .txt files')
parser.add_argument('--output_file', default='', help='output file [default: pred_path/object_detection_evaluation.txt]')
opt = parser.parse_args()

if opt.output_file == '':
    opt.output_file = os.path.join(opt.pred_path, 'object_detection_evaluation.txt')

CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
opt.overlaps             = np.array([0.5,0.25])
# minimum region size for evaluation [verts]
opt.min_region_sizes     = np.array( [ 100 ] )
# distance thresholds [m]
opt.distance_threshes    = np.array( [  float('inf') ] )
# distance confidences
opt.distance_confs       = np.array( [ -float('inf') ] )

def compute_averages(aps):
    d_inf = 0
    o50   = np.where(np.isclose(opt.overlaps,0.5))
    o25   = np.where(np.isclose(opt.overlaps,0.25))
    avg_dict = {}
    #avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict['all_ap_50%'] = np.nanmean(aps[ d_inf,:,o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[ d_inf,:,o25])
    avg_dict["classes"]  = {}
    for (li,label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name]             = {}
        #avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap50%"]    = np.average(aps[ d_inf,li,o50])
        avg_dict["classes"][label_name]["ap25%"]    = np.average(aps[ d_inf,li,o25])
    return avg_dict

def print_results(avgs):
    sep     = "" 
    col1    = ":"
    lineLen = 64

    print("")
    print("#"*lineLen)
    line  = ""
    line += "{:<15}".format("what"      ) + sep + col1
    line += "{:>15}".format("AP_50%"    ) + sep
    line += "{:>15}".format("AP_25%"    ) + sep
    print(line)
    print("#"*lineLen)

    for (li,label_name) in enumerate(CLASS_LABELS):
        ap_50o  = avgs["classes"][label_name]["ap50%"]
        ap_25o  = avgs["classes"][label_name]["ap25%"]
        line  = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_50o ) + sep
        line += sep + "{:>15.3f}".format(ap_25o ) + sep
        print(line)

    all_ap_50o  = avgs["all_ap_50%"]
    all_ap_25o  = avgs["all_ap_25%"]

    print("-"*lineLen)
    line  = "{:<15}".format("average") + sep + col1 
    line += "{:>15.3f}".format(all_ap_50o)  + sep
    line += "{:>15.3f}".format(all_ap_25o)  + sep
    print(line)
    print("")


def write_result_file(avgs, filename):
    _SPLITTER = ','
    with open(filename, 'w') as f:
        f.write(_SPLITTER.join(['class', 'class id', 'ap50', 'ap25']) + '\n')
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            ap50 = avgs["classes"][class_name]["ap50%"]
            ap25 = avgs["classes"][class_name]["ap25%"]
            f.write(_SPLITTER.join([str(x) for x in [class_name, class_id, ap50, ap25]]) + '\n') 

def evaluate(pred_files, gt_files, pred_path, output_file):
    print('evaluating', len(pred_files), 'scans...')
    overlaps = opt.overlaps
    ap_scores = np.zeros( (1, len(CLASS_LABELS) , len(overlaps)) , np.float )
    pred_all = {}
    gt_all = {}
    for i in range(len(pred_files)):
        matches_key = os.path.abspath(gt_files[i])
        image_id = os.path.basename(matches_key)
        # assign gt to predictions
        pred_all[image_id] = []
        gt_all[image_id] = []
        #read prediction file
        lines = open(pred_files[i]).read().splitlines()
        for line in lines:
            parts = line.split(' ')
            if len(parts) != 8:
                util.print_error('invalid object detection prediction file. Expected (per line): [minx] [miny] [minz] [maxx] [maxy] [maxz] [label_id] [score]', user_fault=True)
            bbox = np.array([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])
            class_id =int(float(parts[6]))
            if not class_id in VALID_CLASS_IDS: 
                continue
            classname = ID_TO_LABEL[class_id]
            score = float(parts[7])
            pred_all[image_id].append((classname, bbox, score))
        #read ground truth file 
        lines = open(gt_files[i]).read().splitlines()
        for line in lines:
            parts = line.split(' ')
            if len(parts) != 7:
                util.print_error('invalid object detection ground truth file. Expected (per line): [minx] [miny] [minz] [maxx] [maxy] [maxz] [label_id]', user_fault=True)
            bbox = np.array([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])
            class_id =int(float(parts[6]))
            if not class_id in VALID_CLASS_IDS: 
                continue
            classname = ID_TO_LABEL[class_id]
            gt_all[image_id].append((classname, bbox))
    for oi, overlap_th in enumerate(overlaps): 
        _,_,ap_dict = eval_det(pred_all, gt_all, ovthresh=overlap_th)
        for label in ap_dict:
            id = CLASS_LABELS.index(label)
            ap_scores[0,id, oi] = ap_dict[label]
    #print(ap_scores)
    avgs = compute_averages(ap_scores)
    # print
    print_results(avgs)
    write_result_file(avgs, output_file)


def main():
    pred_files = [f for f in os.listdir(opt.pred_path) if f.endswith('.txt') and f != 'object_detection_evaluation.txt']
    gt_files = []
    if len(pred_files) == 0:
        util.print_error('No result files found.', user_fault=True)
    for i in range(len(pred_files)):
        gt_file = os.path.join(opt.gt_path, pred_files[i])
        if not os.path.isfile(gt_file):
            util.print_error('Result file {} does not match any gt file'.format(pred_files[i]), user_fault=True)
        gt_files.append(gt_file)
        pred_files[i] = os.path.join(opt.pred_path, pred_files[i])

    # evaluate
    evaluate(pred_files, gt_files, opt.pred_path, opt.output_file)


if __name__ == '__main__':
    main()
