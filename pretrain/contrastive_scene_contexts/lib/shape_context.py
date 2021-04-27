import math
import torch
import numpy as np

class ShapeContext(object):
    def __init__(self, r1=0.125, r2=2, nbins_xy=2, nbins_zy=2):
        # right-hand rule
        """
        nbins_xy >= 2
        nbins_zy >= 1
        """
        self.r1 = r1
        self.r2 = r2
        self.nbins_xy = nbins_xy
        self.nbins_zy = nbins_zy
        self.partitions = nbins_xy * nbins_zy * 2
    
    @staticmethod
    def pdist(rel_trans):
        D2 = torch.sum(rel_trans.pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    
    @staticmethod
    def compute_rel_trans(A, B):
        return A.unsqueeze(0) - B.unsqueeze(1)

    @staticmethod
    def hash(A, B, seed):
        '''
        seed = bins of B
        entry < 0 will be ignored
        '''
        mask = (A >= 0) & (B >= 0)
        C = torch.zeros_like(A) - 1
        C[mask] = A[mask] * seed + B[mask]
        return C


    @staticmethod
    def compute_angles(rel_trans):
        """ compute angles between a set of points """
        angles_xy = torch.atan2(rel_trans[:,:,1], rel_trans[:,:,0])
        #angles between 0, 2*pi
        angles_xy = torch.fmod(angles_xy + 2 * math.pi, 2 * math.pi)

        angles_zy = torch.atan2(rel_trans[:,:,1], rel_trans[:,:,2])
        #angles between 0, pi
        angles_zy = torch.fmod(angles_zy + 2 * math.pi, math.pi)
        
        return angles_xy, angles_zy

    def compute_partitions(self, xyz):
        rel_trans = ShapeContext.compute_rel_trans(xyz, xyz)

        # angles
        angles_xy, angles_zy = ShapeContext.compute_angles(rel_trans)
        angles_xy_bins = torch.floor(angles_xy / (2 * math.pi / self.nbins_xy))
        angles_zy_bins = torch.floor(angles_zy / (math.pi / self.nbins_zy))
        angles_bins = ShapeContext.hash(angles_xy_bins, angles_zy_bins, self.nbins_zy)

        # distances
        distance_matrix = ShapeContext.pdist(rel_trans)
        dist_bins = torch.zeros_like(angles_bins) - 1

        # partitions
        mask = (distance_matrix >= self.r1) & (distance_matrix < self.r2)
        dist_bins[mask] = 0
        mask = distance_matrix >= self.r2
        dist_bins[mask] = 1
        
        bins = ShapeContext.hash(dist_bins, angles_bins, self.nbins_xy * self.nbins_zy)
        return bins


    def compute_partitions_fast(self, xyz):
        '''
        fast partitions:  axis-aligned partitions
        '''

        partition_matrix = torch.zeros((xyz.shape[0], xyz.shape[0]))
        partition_matrix = partition_matrix.cuda() -1e9
        distance_matrix = ShapeContext.compute_rel_trans(xyz, xyz)
        maskUp = distance_matrix[:,:,2] > self.r1
        maskDown = distance_matrix[:,:,2] > self.r1
        distance_matrix = ShapeContext.pdist(distance_matrix)

        mask = (distance_matrix[:,:] > self.r1) & (distance_matrix[:,:] <= self.r2)
        partition_matrix[mask & maskUp] = 0
        partition_matrix[mask & maskDown] = 1

        mask = distance_matrix[:,:] > self.r2
        partition_matrix[mask & maskUp] = 2
        partition_matrix[mask & maskDown] = 3
        self.partitions = 4

        return partition_matrix
