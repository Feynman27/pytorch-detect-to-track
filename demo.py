from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv, bbox_transform_inv_legs
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.online_tubes import VideoDataset, VideoPostProcessor 
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

COLOR_WHEEL = ('red', 'blue', 'brown', 'darkblue', 'green',
               'darkgreen', 'brown', 'coral', 'crimson', 'cyan',
               'fuchsia', 'gold', 'indigo', 'red', 'lightblue',
               'lightgreen', 'lime', 'magenta', 'maroon', 'navy',
               'olive', 'orange', 'orangered', 'orchid', 'plum',
               'purple', 'tan', 'teal', 'tomato', 'violet')

fig, ax = None, None

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='imagenet_vid', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='res101',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="output/models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--vid_list', dest='vid_list',
                      help='List of input videos.',
                      nargs='+', required=True)
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def visualize_without_paths(video_dataset, pred_boxes, scores, pred_trk_boxes, det_classes):
    print("Visualizing...")
    list_im = video_dataset._frame_paths 

    num_classes = len(det_classes)
    num_frames = len(list_im)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    for i_frame in range(num_frames-1):
        print('frame: {}/{}'.format(i_frame, num_frames))
        fig, ax = plt.subplots(figsize=(12, 12))
        img_path = list_im[i_frame]
        img = cv2.imread(img_path)
        img = img[:,:,(2,1,0)]
        disp_image = Image.fromarray(np.uint8(img))
        for cls_ind in range(1, num_classes):
            ax.imshow(disp_image, aspect='equal')
            class_name = det_classes[cls_ind]
            keep = torch.nonzero(scores[i_frame][0][:, cls_ind]>CONF_THRESH).view(-1)
            if keep.numel()==0:
                # no detections above threshold for this class
                continue
            cls_scores = scores[i_frame][0][keep][:, cls_ind]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[i_frame][0][keep, :]
            cls_dets = torch.cat([cls_boxes, cls_scores.contiguous().view(-1,1)], dim=1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, 0.3)
            cls_dets = cls_dets[keep.view(-1).long()]
            for ibox in range(cls_dets.size(0)):
                bbox = cls_dets[ibox, :4].cpu().numpy().flatten()
                score = cls_dets[ibox, 4]
                ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor=COLOR_WHEEL[cls_ind], linewidth=3.5)
                        )
                ax.text(bbox[0], bbox[1] - 2,
                        '{:s} {:.3f}'.format(class_name, score),
                        bbox=dict(facecolor=COLOR_WHEEL[cls_ind], alpha=0.5),
                        fontsize=14, color='white')

        # Save image with bboxes overlaid
        plt.axis('off')
        plt.tight_layout()
        #plt.show()
        if not os.path.exists(video_dataset._output_dir):
            os.makedirs(video_dataset._output_dir)
        plt.savefig(os.path.join(video_dataset._output_dir, os.path.basename(img_path)))
        plt.clf()
        plt.close('all')


def visualize_with_paths(video_dataset, video_post_proc):

    print("Visualizing...")
    list_im = video_dataset._frame_paths 
    # define save dir
    save_dir = video_dataset._output_dir
    output_dir = save_dir.replace('.mp4', '')
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    det_classes = video_post_proc.classes
    num_classes = video_post_proc.num_classes 
    num_frames = len(list_im)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()
    #filtered_paths = {cls_name:[] for cls_name in det_classes}
    #for cls_ind in range(1, num_classes):
    #    class_name = det_classes[cls_ind]
    #    smooth_scores = paths[cls_ind]['smooth_scores']
    #    boxes = paths[cls_ind]['boxes']
    #    num_paths = smooth_scores.size(0) # number of paths for this class
    #	for ipath in range(num_paths):
    #        boxes_with_smooth_scores = torch.cat([boxes[ipath],smooth_scores[ipath]], dim=1)
    #        keep = torch.nonzero(boxes_with_smooth_scores[:, -1]>CONF_THRESH).view(-1)
    #        if keep.numel()==0:
                # no detections above threshold for this class
    #            continue
    #        boxes_to_keep = boxes_with_smooth_scores[keep]
            # [x0,y0,x1,y1,score,smooth_score,frame_index]
    #        boxes_to_keep = torch.cat([boxes_to_keep, keep.view(-1,1).float()], dim=1)
    #        filtered_paths[class_name].append(boxes_to_keep)

    for i_frame in range(num_frames):
        print('frame: {}/{}'.format(i_frame, num_frames))
        fig, ax = plt.subplots(figsize=(12, 12))
        img_path = list_im[i_frame]
        img = cv2.imread(img_path)
        img = img[:,:,(2,1,0)]
        disp_image = Image.fromarray(np.uint8(img))
        for i_pth, cls_ind in enumerate(video_post_proc.path_labels): # iterate over path labels
            cls_ind = int(cls_ind)
            ax.imshow(disp_image, aspect='equal')
            class_name = det_classes[cls_ind]
            path_starts =  video_post_proc.path_starts[i_pth]
            path_ends = video_post_proc.path_ends[i_pth]
            if i_frame >= path_starts and i_frame <= path_ends: # is this frame in the current path
                # bboxes for this class path
                bbox = video_post_proc.path_boxes[i_pth][i_frame-path_starts].cpu().numpy() 
                # scores for this class path
                score = video_post_proc.path_scores[i_pth][i_frame-path_starts].cpu().numpy() 
                
                ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor=COLOR_WHEEL[cls_ind], linewidth=3.5)
                        )
                ax.text(bbox[0], bbox[1] - 2,
                        '{:s} {:.3f}'.format(class_name, score[0]),
                        bbox=dict(facecolor=COLOR_WHEEL[cls_ind], alpha=0.5),
                        fontsize=14, color='white')
                
            #class_paths = filtered_paths[class_name]
            #for path in class_paths:
            #    box_id = torch.nonzero(path[:,-1]==i_frame).long().view(-1)
            #    if box_id.numel()==0:
            #        continue
            #    bbox = path[box_id][:, :4].cpu().numpy().flatten()
            #    score = path[box_id][:, 5].cpu().numpy()
        # Save image with bboxes overlaid
        plt.axis('off')
        plt.tight_layout()
        #plt.show()
        im_save_name = os.path.join(output_dir, os.path.basename(img_path))
        print('Image with bboxes saved to {}'.format(im_save_name))
        plt.savefig(im_save_name)
        plt.clf()
        plt.close('all')

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet_vid":
      args.imdb_name = "imagenet_vid_train"
      args.imdbval_name = "imagenet_vid_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'rfcn_detect_track_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  imagenet_vid_classes = ['__background__',  # always index 0
          'airplane', 'antelope', 'bear', 'bicycle',
          'bird', 'bus', 'car', 'cattle',
          'dog', 'domestic_cat', 'elephant', 'fox',
          'giant_panda', 'hamster', 'horse', 'lion',
          'lizard', 'monkey', 'motorcycle', 'rabbit',
          'red_panda', 'sheep', 'snake', 'squirrel',
          'tiger', 'train', 'turtle', 'watercraft',
          'whale', 'zebra']

  # initilize the network here.
  if args.net == 'res101':
    RFCN = resnet(imagenet_vid_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  RFCN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  RFCN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')

  # pdb.set_trace()

  print("load checkpoint %s" % (load_name))

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    RFCN.cuda()

  RFCN.eval()

  start = time.time()
  #max_per_image = 100
  thresh = 0.05
  vis = True
  # legs in siamese net
  n_legs=2
  video_dataset = VideoDataset(args.vid_list, imagenet_vid_classes) 
  # Iterate over each video in the dataset
  for ivid in range(len(video_dataset)):
      # reset
      vid_id = os.path.basename(video_dataset.video_paths[ivid]).replace('.mp4','')
      vid_pred_boxes = [] # container for predicted boxes over all frames
      vid_pred_trk_boxes = [] # container for predicted tracking boxes over all frames
      vid_scores = [] # container for box scores across all frames
      vid_blob = video_dataset[ivid]
      assert os.path.exists(video_dataset.video_name), \
              "File {} does not exist. Confirm input is full path.".format(vid)
      # Iterate over all frame pairs in the ividth video
      start_det = time.time()
      for frames in vid_blob:
          im_data.data.resize_(frames['data'].size()).copy_(frames['data'])
          im_info.data.resize_(frames['im_info'].size()).copy_(frames['im_info'])
          gt_boxes.data.resize_(1, n_legs, 1, 5).zero_()
          num_boxes.data.resize_(1, n_legs, 1).zero_()

          # Add batch dim
          im_data.unsqueeze_(0)
          im_info.unsqueeze_(0)
          n_legs = im_data.size(1)
          batch_size = im_data.size(0)

          rois, cls_prob, bbox_pred, tracking_pred, \
          rpn_loss_cls, rpn_loss_box, \
          RCNN_loss_cls, RCNN_loss_bbox, \
          rois_label, tracking_loss_bbox = RFCN(im_data, im_info, gt_boxes, num_boxes)

          scores = cls_prob.data
          boxes = rois.data[:,:,:,1:5]
          # Track boxes are defined as rois from first frame in pair
          trk_boxes = rois.data[0,:,:,1:5]
          if cfg.TEST.BBOX_REG:
              box_deltas = bbox_pred.data
              if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                  if args.class_agnostic:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                      box_deltas = box_deltas.view(n_legs, batch_size, -1, 4)
                  else:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                      box_deltas = box_deltas.view(n_legs, batch_size, -1, 4*len(imagenet_vid_classes))
              pred_boxes = bbox_transform_inv_legs(boxes, box_deltas, batch_size)
              pred_boxes = clip_boxes(pred_boxes, im_info.data, batch_size)
          else:
              # Simply repeat the boxes, once for each class
              raise NotImplementedError
          trk_box_deltas = tracking_pred.unsqueeze(0).data
          #TODO Check whether this is necessary
          trk_box_deltas = trk_box_deltas.view(-1,4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                  + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
          trk_box_deltas = trk_box_deltas.view(1,-1,4)
          pred_trk_boxes = bbox_transform_inv(trk_boxes, trk_box_deltas, 1)
          pred_trk_boxes = clip_boxes(pred_trk_boxes, im_info.permute(1,0,2)[0].data, 1)

          # Assume scales are same for frames in the same video
          im_scale = im_info.data.squeeze(0)[0][-1]
          #im_scales = im_info[:,:,2].data.contiguous().view(1,-1,1,1).permute(1,0,2,3)
          pred_boxes /= im_scale
          pred_trk_boxes /= im_scale
          # squeeze batch dim
          #scores = scores.squeeze(1) 
          #pred_boxes = pred_boxes.squeeze(1)
          #pred_trk_boxes = pred_trk_boxes.squeeze(0)
          # Permute such that we have (frame_sample_id, n_legs, n_boxes, ...)
          pred_boxes = pred_boxes.permute(1,0,2,3).contiguous()
          scores = scores.permute(1,0,2,3).contiguous()
          vid_pred_boxes.append(pred_boxes)
          vid_scores.append(scores)
          vid_pred_trk_boxes.append(pred_trk_boxes)
          curr_frame_t0 = frames['frame_number'].squeeze()[0]
          curr_frame_t1 = frames['frame_number'].squeeze()[1]
          print("Processed frame pair : t={}, t+tau={} / {}"\
                  .format(curr_frame_t0, curr_frame_t1, video_dataset._n_frames-1))
      end_det = time.time()
      print('Done with detection. Took {} sec'.format(end_det-start_det))
      if len(vid_pred_boxes)==0:
          print("WARNING: No boxes predicted. Make sure your fps is high enough.")
      else:
	  vid_pred_boxes = torch.cat(vid_pred_boxes, dim=0)
	  vid_pred_trk_boxes = torch.cat(vid_pred_trk_boxes, dim=0)
	  vid_scores = torch.cat(vid_scores, dim=0)

	  vid_post_proc = VideoPostProcessor(vid_pred_boxes, vid_scores, 
                                             vid_pred_trk_boxes, imagenet_vid_classes, vid_id)
	  paths = vid_post_proc.class_paths(path_score_thresh=0.5)
	  visualize_with_paths(video_dataset, vid_post_proc)
	  #visualize_without_paths(video_dataset, vid_pred_boxes, vid_scores, vid_pred_trk_boxes, imagenet_vid_classes)




