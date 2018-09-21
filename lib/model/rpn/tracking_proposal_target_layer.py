from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from ..utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch
import pdb

class _TrackingProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_TrackingProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, gt_boxes, num_boxes):

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)
        # Use ground-truth boxes with frame correspondence as the set of candidate rois
        #gt_rois = gt_boxes.new(gt_boxes.size()).zero_()
        #gt_rois[:,:,:,1:5] = gt_boxes[:,:,:,:4]
        #gt_trk_ids = gt_boxes[:,:,:,5]

        
        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image
        labels, rois, bbox_targets, bbox_inside_weights = self._sample_gt_rois_pytorch(
                gt_boxes, fg_rois_per_image, rois_per_image, self._num_classes, num_boxes)

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_tracking_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):
        """Tracking Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of tracking regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights


    def _compute_targets_pytorch(self, gt_rois_t0, gt_rois_t1):
        """Compute tracking regression targets for an image."""

        assert gt_rois_t0.size(1) == gt_rois_t1.size(1)
        assert gt_rois_t0.size(2) == 4
        assert gt_rois_t1.size(2) == 4

        batch_size = gt_rois_t0.size(0)
        rois_per_image = gt_rois_t0.size(1)

        targets = bbox_transform_batch(gt_rois_t0, gt_rois_t1)

        # TODO Check if we need this step
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets


    def _get_track_correspondence(self, gt_boxes, num_boxes):
        """Check whether gt track in frame t has correspondence in frame t+tau
        """
        n_twins = gt_boxes.size(0)
        batch_size = gt_boxes.size(1)
        correspondence_matrices = []
        for i_leg in range(n_twins-1):
            for j_leg in range(i_leg+1, n_twins):
                for i_batch in range(batch_size):
                    i_num_boxes = num_boxes[i_leg][i_batch][0]
                    j_num_boxes = num_boxes[j_leg][i_batch][0]
                    if j_num_boxes==0 or i_num_boxes==0:
                        padded_corr_matrix = torch.zeros(
                            gt_boxes[i_leg][i_batch].size(0),gt_boxes[i_leg][i_batch].size(0)).type_as(gt_boxes)
                        padded_corr_matrix = padded_corr_matrix.long()

                    else:
                        twin2_trk_id_set = gt_boxes[j_leg][i_batch][:j_num_boxes, 5] 
                        twin1_trk_id_set = gt_boxes[i_leg][i_batch][:i_num_boxes, 5]
                        # Create N_t+tau by N_t correspondence matrix
                        X = twin2_trk_id_set.expand(twin1_trk_id_set.size(0), twin2_trk_id_set.size(0)).t()
                        # transpose to N_t by N_t+tau matrix
                        corr_matrix = (twin1_trk_id_set==X).t()
                        # fatten up with zeros again
                        padding = gt_boxes.size(2)-i_num_boxes
                        padded_corr_matrix = torch.zeros(
                                gt_boxes[i_leg][i_batch].size(0),
                                gt_boxes[i_leg][i_batch].size(0)).type_as(corr_matrix)
                        padded_corr_matrix[:i_num_boxes, :j_num_boxes] = corr_matrix
                        padded_corr_matrix = padded_corr_matrix.long()
                    correspondence_matrices.append(padded_corr_matrix.unsqueeze(0))
        batch_corr_matrices = torch.cat(correspondence_matrices, dim=0)
        return batch_corr_matrices

    def _sample_gt_rois_pytorch(self, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, num_boxes):
        """Generate sample of RoIs comprised of ground-truth rois.
        """
        n_twins = gt_boxes.size(0)
        batch_size = gt_boxes.size(1)
        num_boxes_per_img = gt_boxes.size(2)
        # N_t by N_t+tau correspondence matrix
        trk_correspondences = self._get_track_correspondence(gt_boxes, num_boxes)
        batch_gt_rois_t0 = gt_boxes.new(batch_size, num_boxes_per_img, 6).zero_()
        batch_gt_rois_t1 = batch_gt_rois_t0.clone()
        labels = gt_boxes[:,:,:,4]
        tracking_labels_batch = labels.new(batch_size, num_boxes_per_img).zero_()
        tracking_rois_batch = gt_boxes.new(batch_size, num_boxes_per_img, 5).zero_()
        #batch_gt_rois_t0=[]
        #batch_gt_rois_t1=[]
        for i_bch in range(batch_size):
            row_inds = torch.nonzero(trk_correspondences[i_bch].sum(dim=1)).long().view(-1)
            col_inds = torch.nonzero(trk_correspondences[i_bch].sum(dim=0)).long().view(-1)
            gt_boxes_t0 = gt_boxes[0][i_bch]
            gt_boxes_t1 = gt_boxes[1][i_bch]
            if row_inds.numel()>0 and col_inds.numel()>0:
                # gt rois with correspondence across frames
                # TODO handle case where row_inds and/or col_inds are empty
                # Probably easiest just to filter from roidb
                gt_rois_t0 = torch.index_select(gt_boxes_t0, 0, row_inds)
                gt_rois_t1 = torch.index_select(gt_boxes_t1, 0, col_inds)
                # align tracks across time frames
                _, sorted_gt_inds =  torch.sort(gt_rois_t0[:, 5], descending=False)
                gt_rois_t0 = gt_rois_t0[sorted_gt_inds]
                _, sorted_gt_inds =  torch.sort(gt_rois_t1[:, 5], descending=False)
                gt_rois_t1 = gt_rois_t1[sorted_gt_inds]
                assert gt_rois_t0.size(0)==gt_rois_t1.size(0), \
                        "[tracking_proposal_target_layer] gt rois dim are not equal."
                temp_num_rois_t0 = gt_rois_t0.size(0) 
                temp_num_rois_t1 = gt_rois_t1.size(0) 
                batch_gt_rois_t0[i_bch][:temp_num_rois_t0] = gt_rois_t0
                batch_gt_rois_t1[i_bch][:temp_num_rois_t1] = gt_rois_t1
                tracking_labels_batch[i_bch] = batch_gt_rois_t0[i_bch][:, 4] # uncomment this line!
                tracking_rois_batch[i_bch][:,0] = i_bch
                tracking_rois_batch[i_bch][:,1:] = gt_boxes[0][i_bch][:,:4]

        tracking_target_data = self._compute_targets_pytorch(batch_gt_rois_t0[:,:,:4], 
                batch_gt_rois_t1[:,:,:4])
        tracking_targets, tracking_inside_weights = \
                self._get_tracking_regression_labels_pytorch(tracking_target_data, 
                        tracking_labels_batch, num_classes)
        # set tracking rois to gt rois in frame t0
        #tracking_rois_batch = gt_boxes[0][:,:,:5]
        return tracking_labels_batch, tracking_rois_batch, tracking_targets, tracking_inside_weights
