# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Utility functions for metric evaluation.

Author: Or Litany and Charles R. Qi
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np

def calc_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths        
    Returns:
        iou
    """        
        
    max_a = box_a[3:6]
    max_b = box_b[3:6]
    min_max = np.array([max_a, max_b]).min(0)
        
    min_a = box_a[0:3]
    min_b = box_b[0:3]
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = (box_a[3:6] - box_a[0:3]).prod()
    vol_b = (box_b[3:6] - box_b[0:3]).prod()
    union = vol_a + vol_b - intersection
    return 1.0*intersection / union



