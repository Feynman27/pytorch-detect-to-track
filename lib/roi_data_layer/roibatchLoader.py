
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch, get_minibatch
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb

_DEBUG = False

class roibatchLoader(data.Dataset):
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
    self._roidb = roidb
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.normalize = normalize
    self.batch_size = batch_size
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.data_size = len(self.ratio_list)
    # given the ratio_list, we want to make the ratio same for each batch.
    self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
    num_batch = int(np.ceil(len(ratio_index) / batch_size))
    for i in range(num_batch):
        left_idx = i*batch_size
        right_idx = min((i+1)*batch_size-1, self.data_size-1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1


        self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio

  def _plot_image(self, data, gt_boxes, num_boxes):
      import matplotlib.pyplot as plt
      X=data.cpu().numpy().copy()
      X += cfg.PIXEL_MEANS
      X = X.astype(np.uint8) 
      X = X.squeeze(0)
      boxes = gt_boxes.squeeze(0)[:num_boxes.view(-1)[0],:].cpu().numpy().copy()

      fig, ax = plt.subplots(figsize=(8,8))
      ax.imshow(X[:,:,::-1], aspect='equal')
      for i in range(boxes.shape[0]):
          bbox = boxes[i, :4]
          ax.add_patch(
                  plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2]-bbox[0],
                                 bbox[3]-bbox[1], fill=False, linewidth=2.0)
                  )
      #plt.imshow(X[:,:,::-1])
      plt.tight_layout()
      plt.show()

  def __getitem__(self, index):
    if self.training:
        index_ratio = int(self.ratio_index[index])
    else:
        index_ratio = index
    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    #index = 32014 # temp hack for testing case where crop excluded gt boxes
    minibatch_db = self._roidb[index_ratio] 
    blobs=[]
    data = []
    padding_data=[]
    im_info = []
    data_heights = []
    data_widths = []
    gt_boxes = []
    gt_boxes_padding = []
    num_boxes = []
    # check for duplicate tracks within same frame
    assert len(minibatch_db[0]['track_id']) == len(np.unique(minibatch_db[0]['track_id'])), \
            'Cannot have >1 track with same id in same frame.'
    assert len(minibatch_db[1]['track_id']) == len(np.unique(minibatch_db[1]['track_id'])), \
            'Cannot have >1 track with same id in same frame.'

    # Iterate through each entry in the sample tuple 
    for ientry, entry in enumerate(minibatch_db):
        blobs.append(get_minibatch([entry], self._num_classes))
        data.append(torch.from_numpy(blobs[ientry]['data']))
        im_info.append(torch.from_numpy(blobs[ientry]['im_info']))
        data_heights.append(data[ientry].size(1)) 
        data_widths.append(data[ientry].size(2)) 
        # random shuffle the bounding boxes
        #np.random.shuffle(blobs[ientry]['gt_boxes'])
        if not self.training and blobs[ientry]['gt_boxes'].shape[0]==0:
          blobs[ientry]['gt_boxes'] = np.ones((1,6), dtype=np.float32)
        gt_boxes.append(torch.from_numpy(blobs[ientry]['gt_boxes']))
        if self.training:
            ########################################################
            # padding the input image to fixed size for each group #
            ########################################################
            # if the image needs to be cropped, crop to the target size
            ratio = self.ratio_list_batch[index]
            if self._roidb[index_ratio][0]['need_crop']:
                if ratio < 1.:
                    # this means that data_width << data_height and we crop the height
                    min_y = int(torch.min(gt_boxes[ientry][:,1]))
                    max_y = int(torch.max(gt_boxes[ientry][:,3]))
                    trim_size = int(np.floor(data_widths[ientry] / ratio))
                    if trim_size > data_heights[ientry]:
                        trim_size = data_heights[ientry] 
                    box_region = max_y - min_y + 1
                    if min_y==0:
                        y_s = 0
                    else:
                        if (box_region-trim_size) < 0:
                            y_s_min = max(max_y-trim_size, 0)
                            y_s_max = min(min_y, data_heights[ientry]-trim_size)
                            if y_s_min == y_s_max:
                                y_s = y_s_min
                            else:
                                y_s = np.random.choice(range(y_s_min, y_s_max))
                        else:
                            y_s_add = int((box_region-trim_size)/2)
                            if y_s_add == 0:
                                y_s = min_y
                            else:
                                y_s = np.random.choice(range(min_y, min_y+y_s_add))
                    # crop the image
                    data[ientry] = data[ientry][:, y_s:(y_s + trim_size), :, :]
                    # shift y coordiante of gt_boxes
                    gt_boxes[ientry][:, 1] = gt_boxes[ientry][:, 1] - float(y_s)
                    gt_boxes[ientry][:, 3] = gt_boxes[ientry][:, 3] - float(y_s)
                    # update gt bounding box according to trim
                    gt_boxes[ientry][:, 1].clamp_(0, trim_size - 1)
                    gt_boxes[ientry][:, 3].clamp_(0, trim_size - 1)
                else:
                    # data_width >> data_height so crop width
                    min_x = int(torch.min(gt_boxes[ientry][:,0]))
                    max_x = int(torch.max(gt_boxes[ientry][:,2]))
                    trim_size = int(np.ceil(data_heights[ientry] * ratio))
                    if trim_size > data_widths[ientry]:
                        trim_size = data_widths[ientry]
                    box_region = max_x - min_x + 1
                    if min_x == 0:
                        x_s = 0
                    else:
                        if (box_region-trim_size) < 0:
                            x_s_min = max(max_x-trim_size, 0)
                            x_s_max = min(min_x, data_widths[ientry]-trim_size)
                            if x_s_min == x_s_max:
                                x_s = x_s_min
                            else:
                                x_s = np.random.choice(range(x_s_min, x_s_max))
                        else:
                            x_s_add = int((box_region-trim_size)/2)
                            if x_s_add == 0:
                                x_s = min_x
                            else:
                                x_s = np.random.choice(range(min_x, min_x+x_s_add))
                    # crop the image
                    data[ientry] = data[ientry][:, :, x_s:(x_s + trim_size), :]

                    # shift x coordiante of gt_boxes[ientry]
		    gt_boxes[ientry][:, 0] = gt_boxes[ientry][:, 0] - float(x_s)
		    gt_boxes[ientry][:, 2] = gt_boxes[ientry][:, 2] - float(x_s)
		    # update gt bounding box according the trip
		    gt_boxes[ientry][:, 0].clamp_(0, trim_size - 1)
		    gt_boxes[ientry][:, 2].clamp_(0, trim_size - 1)
            # based on the ratio, pad the image.
            if ratio < 1:
                # data_width < data_height
                trim_size = int(np.floor(data_widths[ientry] / ratio))
                padding_data.append(torch.FloatTensor(int(np.ceil(data_widths[ientry] / ratio)),\
                        data_widths[ientry], 3).zero_())
                padding_data[ientry][:data_heights[ientry], :, :] = data[ientry][0]
                im_info[ientry][0,0] = padding_data[ientry].size(0)
            elif ratio > 1:
                # data_width > data_height
                padding_data.append(torch.FloatTensor(data_heights[ientry],\
                        int(np.ceil(data_heights[ientry] * ratio)), 3).zero_())
                padding_data[ientry][:, :data_widths[ientry], :] = data[ientry][0]
                im_info[ientry][0,1] = padding_data[ientry].size(1)
            else:
                trim_size = min(data_heights[ientry], data_widths[ientry])
                padding_data.append(torch.FloatTensor(trim_size, trim_size, 3).zero_())
                padding_data[ientry] = data[ientry][0][:trim_size, :trim_size, :]
                # gt_boxes[ientry].clamp_(0, trim_size)
                gt_boxes[ientry][:, :4].clamp_(0, trim_size)
                im_info[ientry][0, 0] = trim_size
                im_info[ientry][0, 1] = trim_size
            # check the bounding box:
            not_keep = (gt_boxes[ientry][:,0] \
                    == gt_boxes[ientry][:,2]) | (gt_boxes[ientry][:,1] == gt_boxes[ientry][:,3])
            keep = torch.nonzero(not_keep == 0).view(-1)

            gt_boxes_padding.append(torch.FloatTensor(self.max_num_box, gt_boxes[ientry].size(1)).zero_())
            if keep.numel() != 0:
                gt_boxes[ientry] = gt_boxes[ientry][keep]
                num_boxes.append(torch.LongTensor([min(gt_boxes[ientry].size(0), self.max_num_box)]).cuda())
                curr_num_boxes = int(num_boxes[ientry][0])
                gt_boxes_padding[ientry][:curr_num_boxes,:] = gt_boxes[ientry][:curr_num_boxes]
            else:
                num_boxes.append(torch.LongTensor(1).cuda().zero_())

            # permute trim_data to adapt to downstream processing
            padding_data[ientry] = padding_data[ientry].squeeze(0).permute(2, 0, 1).contiguous()
            padding_data[ientry] = padding_data[ientry].unsqueeze(0)
            #im_info[ientry] = im_info[ientry].view(3)
            gt_boxes_padding[ientry] = gt_boxes_padding[ientry].unsqueeze(0)
            num_boxes[ientry] = num_boxes[ientry].unsqueeze(0)

            #return padding_data, im_info, gt_boxes_padding, num_boxes
        else:
            data[ientry] = data[ientry].permute(0, 3, 1, 2).contiguous().\
                    view(3, data_heights[ientry], data_widths[ientry])
            data[ientry] = data[ientry].unsqueeze(0)
            #im_info[ientry] = im_info[ientry].view(3)

            #gt_boxes.append(torch.FloatTensor([1,1,1,1,1]))
            gt_boxes_padding.append(torch.FloatTensor(self.max_num_box, gt_boxes[ientry].size(1)).zero_())
            #gt_boxes[ientry] = gt_boxes[ientry].unsqueeze(0)
            num_boxes.append(torch.LongTensor([min(gt_boxes[ientry].size(0), self.max_num_box)]).cuda())
            #num_boxes.append(torch.LongTensor(1).cuda().zero_())
            num_boxes[ientry] = num_boxes[ientry].unsqueeze(0)
            curr_num_boxes = int(num_boxes[ientry][0])
            gt_boxes_padding[ientry][:curr_num_boxes,:] = gt_boxes[ientry][:curr_num_boxes]
            gt_boxes_padding[ientry] = gt_boxes_padding[ientry].unsqueeze(0)

            #return data, im_info, gt_boxes, num_boxes
        if _DEBUG:
            if self.training:
                print(gt_boxes_padding[ientry])
		print(padding_data[ientry].size())
		self._plot_image(padding_data[ientry].permute(0,2,3,1), gt_boxes_padding[ientry], num_boxes[ientry])
            else:
                print(gt_boxes[ientry])
		print(data[ientry].size())
		self._plot_image(data[ientry].permute(0,2,3,1), gt_boxes[ientry], num_boxes[ientry])

    im_info_pair = torch.cat(im_info, dim=0)
    num_boxes = torch.cat(num_boxes, dim=0)
    if self.training:
        data_pair = torch.cat(padding_data, dim=0)
        gt_boxes_padding_pair = torch.cat(gt_boxes_padding, dim=0)
        return data_pair, im_info_pair, gt_boxes_padding_pair, num_boxes
    else:
        data_pair = torch.cat(data, dim=0)
        gt_boxes_padding_pair = torch.cat(gt_boxes_padding, dim=0)
        #gt_boxes = torch.cat(gt_boxes, dim=0)
        return data_pair, im_info_pair, gt_boxes_padding_pair, num_boxes

  def __len__(self):
    return len(self._roidb)
