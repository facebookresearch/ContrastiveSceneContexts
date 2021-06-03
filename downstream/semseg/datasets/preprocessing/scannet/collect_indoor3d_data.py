import os
import sys
import plyfile
import json
import time
import torch
import argparse
import numpy as np

def get_raw2scannet_label_map():
    lines = [line.rstrip() for line in open('scannetv2-labels.combined.tsv')]
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        elements = lines[i].split('\t')
        # raw_name = elements[0]
        # nyu40_name = elements[6]
        raw_name = elements[1]
        nyu40_id = elements[4]
        nyu40_name = elements[7]
        raw2scannet[raw_name] = nyu40_id
    return raw2scannet
g_raw2scannet = get_raw2scannet_label_map()
RAW2SCANNET = g_raw2scannet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/canis/Datasets/ScanNet/public/v2/scans/')
    parser.add_argument('--output', default='./output')
    opt = parser.parse_args()
    return opt

def main(config):
    for scene_name in os.listdir(config.input):
        print(scene_name)
        # Over-segmented segments: maps from segment to vertex/point IDs
        segid_to_pointid = {}
        segfile = os.path.join(config.input, scene_name, '%s_vh_clean_2.0.010000.segs.json'%(scene_name))
        with open(segfile) as jsondata:
            d = json.load(jsondata)
            seg = d['segIndices']
        for i in range(len(seg)):
            if seg[i] not in segid_to_pointid:
                segid_to_pointid[seg[i]] = []
            segid_to_pointid[seg[i]].append(i)
        
        # Raw points in XYZRGBA
        ply_filename = os.path.join(config.input, scene_name, '%s_vh_clean_2.ply' % (scene_name))
        f = plyfile.PlyData().read(ply_filename)
        points = np.array([list(x) for x in f.elements[0]])
        
        # Instances over-segmented segment IDs: annotation on segments
        instance_segids = []
        labels = []
        annotation_filename = os.path.join(config.input, scene_name, '%s.aggregation.json'%(scene_name))
        with open(annotation_filename) as jsondata:
            d = json.load(jsondata)
            for x in d['segGroups']:
                instance_segids.append(x['segments'])
                labels.append(x['label'])

        
        # Each instance's points
        instance_labels = np.zeros(points.shape[0])
        semantic_labels = np.zeros(points.shape[0])
        for i in range(len(instance_segids)):
            segids = instance_segids[i]
            pointids = []
            for segid in segids:
                pointids += segid_to_pointid[segid]
            pointids = np.array(pointids)
            instance_labels[pointids] = i+1
            semantic_labels[pointids] = RAW2SCANNET[labels[i]]
           
        colors = points[:,3:6]
        points = points[:,0:3] # XYZ+RGB+NORMAL
        torch.save((points, colors, semantic_labels, instance_labels), os.path.join(config.output, scene_name+'.pth'))

if __name__=='__main__':
    config = parse_args()
    os.makedirs(config.output, exist_ok=True)
    main(config)
