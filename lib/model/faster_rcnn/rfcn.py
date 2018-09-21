import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.correlation.modules.correlation import Correlation
from model.rpn.rpn import _RPN
#from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.psroi_pooling.modules.psroi_pool import _PSRoIPooling
#from model.roi_crop.modules.roi_crop import _RoICrop
#from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.tracking_proposal_target_layer import _TrackingProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _RFCN(nn.Module):
    """ RFCN """
    def __init__(self, classes, class_agnostic):
        super(_RFCN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_reg_classes = (1 if class_agnostic else len(classes))
        self.class_agnostic = class_agnostic
        self.n_bbox_reg = (4 if class_agnostic else len(classes))
        # loss
        self.RFCN_loss_cls = 0
        self.RFCN_loss_bbox = 0

        # define rpn
        self.RFCN_rpn = _RPN(self.dout_base_model)
        self.RFCN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RFCN_tracking_proposal_target = _TrackingProposalTargetLayer(self.n_classes)
        #self.RFCN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RFCN_psroi_cls_pool = _PSRoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 
                                spatial_scale=1.0/16.0, group_size=7, output_dim=self.n_classes)
        self.RFCN_psroi_loc_pool = _PSRoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 
                                spatial_scale=1.0/16.0, group_size=7, output_dim=4*self.n_reg_classes)
        #self.RFCN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        #self.RFCN_roi_crop = _RoICrop()

	self.RFCN_cls_net = nn.Conv2d(512,self.n_classes*7*7, [1,1], padding=0, stride=1)
        nn.init.normal(self.RFCN_cls_net.weight.data, 0.0, 0.01)
        
	self.RFCN_bbox_net = nn.Conv2d(512, 4*self.n_reg_classes*7*7, [1,1], padding=0, stride=1)
	nn.init.normal(self.RFCN_bbox_net.weight.data, 0.0, 0.01)

	#self.corr_bbox_net = nn.Conv2d(1051, 4*self.n_reg_classes*7*7, [1,1], padding=0, stride=1)
	#nn.init.normal(self.corr_bbox_net.weight.data, 0.0, 0.01)

	self.conv3_corr_layer = Correlation(pad_size=8, kernel_size=1, max_displacement=8, stride1=2, stride2=2)
	self.conv4_corr_layer = Correlation(pad_size=8, kernel_size=1, max_displacement=8, stride1=1, stride2=1)
	self.conv5_corr_layer = Correlation(pad_size=8, kernel_size=1, max_displacement=8, stride1=1, stride2=1) 

        self.RFCN_cls_score = nn.AvgPool2d((7,7), stride=(7,7))
        self.RFCN_bbox_pred = nn.AvgPool2d((7,7), stride=(7,7))
        self.RFCN_tracking_pred = nn.AvgPool2d((7,7), stride=(7,7))

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        # permute such that we have (n_legs, batch_size, C, H, W)
        im_data = im_data.permute(1,0,2,3,4).contiguous()
        im_info = im_info.permute(1,0,2).contiguous()
        gt_boxes = gt_boxes.permute(1,0,2,3).contiguous()
        num_boxes = num_boxes.permute(1,0,2).contiguous()

        batch_size = im_data.size(1)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        num_legs = im_data.size(0)
        base_conv3 = [] 
        base_conv4 = [] 
        base_conv5 = [] 
        base_feat = [] 
        rois=[]
        rois_label=[]
        rpn_loss_cls=[]
        rpn_loss_bbox=[]
        rfcn_cls=[]
        rfcn_bbox=[]
        cls_score=[]
        cls_prob=[]
        bbox_pred=[]
        RFCN_loss_cls=[]
        RFCN_loss_bbox=[]
        # Iterate over each leg of the siamese net
        for i_leg in range(num_legs):
            leg_im_data = im_data[i_leg]
            # feed image data to base model to obtain base feature map
            # output of feature map will be (batch_size, C, H, W)
            base_feats = self._im_to_head(leg_im_data)
            base_conv3.append(base_feats[0])
            base_conv4.append(base_feats[1])
            base_conv5.append(base_feats[2])
            base_feat.append(base_feats[3])
            
            rfcn_cls.append(self.RFCN_cls_net(base_feat[i_leg]))
            rfcn_bbox.append(self.RFCN_bbox_net(base_feat[i_leg]))
            # feed base feature map tp RPN to obtain rois
            rpn_out = self.RFCN_rpn(base_feat[i_leg], 
                                    im_info[i_leg], 
                                    gt_boxes[i_leg][:,:,:5], 
                                    num_boxes[i_leg])
            rois.append(rpn_out[0]) 
            rpn_loss_cls.append(rpn_out[1]) 
            rpn_loss_bbox.append(rpn_out[2])
            if self.training:
                leg_roi_data = self.RFCN_proposal_target(rois[i_leg], gt_boxes[i_leg][:,:,:5], num_boxes[i_leg])
                rois[i_leg], leg_rois_label, \
                        leg_rois_target, leg_rois_inside_ws, leg_rois_outside_ws = leg_roi_data

                leg_rois_label = Variable(leg_rois_label.view(-1).long())
                leg_rois_target = Variable(leg_rois_target.view(-1, leg_rois_target.size(2)))
                leg_rois_inside_ws = Variable(leg_rois_inside_ws.view(-1, leg_rois_inside_ws.size(2)))
                leg_rois_outside_ws = Variable(leg_rois_outside_ws.view(-1, leg_rois_outside_ws.size(2)))
                rois_label.append(leg_rois_label)
            else:
                leg_rois_label = None
                leg_rois_target = None
                leg_rois_inside_ws = None
                leg_rois_outside_ws = None
                rpn_loss_cls[i_leg] = Variable(torch.zeros(1).cuda(), volatile=True)
                rpn_loss_bbox[i_leg] = Variable(torch.zeros(1).cuda(), volatile=True)
            rois[i_leg] = Variable(rois[i_leg])
            leg_pooled_cls_feat = self.RFCN_psroi_cls_pool(rfcn_cls[i_leg], rois[i_leg].view(-1,5))
            leg_pooled_loc_feat = self.RFCN_psroi_loc_pool(rfcn_bbox[i_leg], rois[i_leg].view(-1,5))
            # compute object classification probability
            cls_score.append(self.RFCN_cls_score(leg_pooled_cls_feat).squeeze())
            cls_prob.append(F.softmax(cls_score[i_leg], dim=1))

            # compute bbox offset
            bbox_pred.append(self.RFCN_bbox_pred(leg_pooled_loc_feat).squeeze())

            if self.training and not self.class_agnostic:
                # select the corresponding columns according to roi labels
                bbox_pred_view = bbox_pred[i_leg].view(bbox_pred[i_leg].size(0), 
                        int(bbox_pred[i_leg].size(1) / 4), 4)
                bbox_pred_select = torch.gather(bbox_pred_view, 1, 
                        leg_rois_label.view(leg_rois_label.size(0), 1, 1).expand(leg_rois_label.size(0), 1, 4))
                bbox_pred[i_leg] = bbox_pred_select.squeeze(1)

            RFCN_loss_cls.append(torch.zeros(1).cuda())
            RFCN_loss_bbox.append(torch.zeros(1).cuda())

            if self.training:
                # classification loss
                RFCN_loss_cls[i_leg] = F.cross_entropy(cls_score[i_leg], leg_rois_label)
                # bounding box regression L1 loss
                RFCN_loss_bbox[i_leg] = _smooth_l1_loss(bbox_pred[i_leg], 
                    leg_rois_target, leg_rois_inside_ws, leg_rois_outside_ws)


            cls_prob[i_leg] = cls_prob[i_leg].view(batch_size, rois[i_leg].size(1), -1)
            bbox_pred[i_leg] = bbox_pred[i_leg].view(batch_size, rois[i_leg].size(1), -1)


        # calculate correlations
        tracking_feature_list = []
        tracking_feature_list += rfcn_bbox 
        for i_leg in range(num_legs-1):
            for j_leg in range(i_leg+1, num_legs):
                this_corr3 = self.conv3_corr_layer(base_conv3[i_leg], base_conv3[j_leg])
                this_corr4 = self.conv4_corr_layer(base_conv4[i_leg], base_conv4[j_leg])
                this_corr5 = self.conv5_corr_layer(base_conv5[i_leg], base_conv5[j_leg])
                tracking_feature_list += [this_corr3, this_corr4, this_corr5]
        tracking_feat = torch.cat(tracking_feature_list, dim=1)
        tracking_reg_coords = self.corr_bbox_net(tracking_feat)

        if self.training:
            tracking_roi_data = self.RFCN_tracking_proposal_target(gt_boxes, num_boxes)
            tracking_rois, tracking_rois_label, \
                    tracking_rois_target, tracking_rois_inside_ws, tracking_rois_outside_ws = tracking_roi_data
            tracking_rois_label = Variable(tracking_rois_label.view(-1).long())
            tracking_rois_target = Variable(tracking_rois_target.view(-1, tracking_rois_target.size(2)))
            tracking_rois_inside_ws = Variable(tracking_rois_inside_ws.view(-1, tracking_rois_inside_ws.size(2)))
            tracking_rois_outside_ws = Variable(tracking_rois_outside_ws.view(-1, tracking_rois_outside_ws.size(2)))
            tracking_rois = Variable(tracking_rois)
        else:
            tracking_rois_label = None
            tracking_rois_target = None
            tracking_rois_inside_ws = None
            tracking_rois_outside_ws = None
            RFCN_tracking_loss_bbox = Variable(torch.zeros(1).cuda(), volatile=True) 
            tracking_rois = rois[0].clone() # set tracking rois to rois in frame t0

        psroi_pooled_tracking_rois = self.RFCN_psroi_loc_pool(tracking_reg_coords, 
                tracking_rois.contiguous().view(-1,5)) 
        tracking_pred = self.RFCN_tracking_pred(psroi_pooled_tracking_rois).squeeze()
        if self.training:
            # get foreground gt roi inds
            gt_roi_inds = torch.nonzero(tracking_rois_label>0).view(-1)
            # bounding box regression L1 loss
            #if gt_roi_inds.numel()>0:
            RFCN_tracking_loss_bbox = _smooth_l1_loss(tracking_pred, 
                    tracking_rois_target, 
                    tracking_rois_inside_ws, 
                    tracking_rois_outside_ws)
            #else:
            #    RFCN_tracking_loss_bbox = Variable(torch.zeros(1).cuda()) 
        # Expand to include leg dim
        # (n_legs,n_batch,n_rows,n_columns)
        rois = map(lambda x:x.unsqueeze(0), rois)
        rois = torch.cat(rois, dim=0)
        cls_prob = map(lambda x:x.unsqueeze(0), cls_prob)
        cls_prob = torch.cat(cls_prob, dim=0)
        bbox_pred = map(lambda x:x.unsqueeze(0), bbox_pred)
        bbox_pred = torch.cat(bbox_pred, dim=0)
        # label of rois (n_legs, n_batches, n_rois)
        if len(rois_label)>0:
            rois_label = map(lambda x:x.unsqueeze(0), rois_label)
            rois_label = torch.cat(rois_label, dim=0)
            rois_label = rois_label.view(2,2,-1)
        # Average loss across batch (n_legs, 1)
        rpn_loss_cls = map(lambda x:x.unsqueeze(0), rpn_loss_cls)
        rpn_loss_cls = torch.cat(rpn_loss_cls, dim=0)
        rpn_loss_bbox = map(lambda x:x.unsqueeze(0), rpn_loss_bbox)
        rpn_loss_bbox = torch.cat(rpn_loss_bbox, dim=0)
        RFCN_loss_cls = map(lambda x:x.unsqueeze(0), RFCN_loss_cls)
        RFCN_loss_cls = torch.cat(RFCN_loss_cls, dim=0)
        RFCN_loss_bbox = map(lambda x:x.unsqueeze(0), RFCN_loss_bbox)
        RFCN_loss_bbox = torch.cat(RFCN_loss_bbox, dim=0)
        #RFCN_tracking_loss_bbox = Variable(torch.zeros(1).cuda()) # temp hack
        # Catch nans in loss
        #if rpn_loss_cls.mean().data[0]!=rpn_loss_cls.mean().data[0] or \
        #        rpn_loss_bbox.mean().data[0]!=rpn_loss_bbox.mean().data[0] or \
        #        RFCN_loss_cls.mean().data[0]!=RFCN_loss_cls.mean().data[0] or \
        #        RFCN_loss_bbox.mean().data[0]!=RFCN_loss_bbox.mean().data[0] or \
        #        RFCN_tracking_loss_bbox.mean().data[0] != RFCN_tracking_loss_bbox.mean().data[0]:
                    #print("rpn_loss_cls: {.4f}, \
                    #        rpn_loss_bbox: {.4f}, \
                    #        RFCN_loss_cls: {.4f}, \
                    #        RFCN_loss_cls: {.4f}, \
                    #        RFCN_tracking_loss_bbox: {.4f}"\
                    #        .format(rpn_loss_cls.mean().data[0], 
                    #            rpn_loss_bbox.mean().data[0], 
                    #            RFCN_loss_cls.mean().data[0], 
                    #            RFCN_loss_bbox.mean().data[0], 
                    #            RFCN_tracking_loss_bbox.mean().data[0]))
        #            import pdb; pdb.set_trace()

        return rois, cls_prob, bbox_pred, tracking_pred, rpn_loss_cls, rpn_loss_bbox, \
                RFCN_loss_cls, RFCN_loss_bbox, rois_label, RFCN_tracking_loss_bbox

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()
        if not self.pretrained_rfcn:
            normal_init(self.RFCN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.RFCN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
            normal_init(self.RFCN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
            #normal_init(self.RFCN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
            #normal_init(self.RFCN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
