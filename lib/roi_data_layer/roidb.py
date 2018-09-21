"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
import numpy as np
from model.utils.config import cfg
from datasets.factory import get_imdb
import PIL
import pdb

def create_roi_pairs(roidb, training, duplicate_frames=False):
  '''
  Use roidb to create pairs of consecutive frames from same video snippet
    :param roidb: an roidb data structure
    :param duplicate_frames: replicate image to form a pair with the same image
    :return: list of tuples
  '''
  if duplicate_frames:
      print("Duplicating frames for each roidb entry.")
      num_entries = len(roidb)
  else:
      num_entries = len(roidb)-1

  roidb_frame_pairs = []
  for ientry in range(num_entries):
      if not duplicate_frames:
          video_snippet1 = roidb[ientry]['video_snippet']
          video_snippet2 = roidb[ientry+1]['video_snippet']
          track_ids1 = set(roidb[ientry]['track_id'])
          track_ids2 = set(roidb[ientry+1]['track_id'])
          num_track_overlaps = len(track_ids1.intersection(track_ids2))
          # make sure we don't match flipped/non-flipped frames
          flips_agree = (roidb[ientry]['flipped']==roidb[ientry+1]['flipped'])

      # entries must come from same snippet and >=1 track must be present in both frames
      if training:
        if duplicate_frames:
          roidb_frame_pairs.append((roidb[ientry], roidb[ientry]))
        elif (video_snippet1==video_snippet2) and (num_track_overlaps>0) and (flips_agree):
          roidb_frame_pairs.append((roidb[ientry], roidb[ientry+1]))
        else:
          continue
      elif video_snippet1==video_snippet2: # frames must come from same video
          roidb_frame_pairs.append((roidb[ientry], roidb[ientry+1]))
      else:
          continue
  print("Pairs in roidb: {}".format(len(roidb_frame_pairs)))
  max_frames = len(roidb)
  assert len(roidb_frame_pairs)<=max_frames, "Something is wrong. Too many frame pairs."
  return roidb_frame_pairs

def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """

  roidb = imdb.roidb
  if not (imdb.name.startswith('coco')):
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
         for i in range(imdb.num_images)]
         
  for i in range(len(imdb.image_index)):
    roidb[i]['img_id'] = imdb.image_id_at(i)
    roidb[i]['image'] = imdb.image_path_at(i)
    if not (imdb.name.startswith('coco')):
      roidb[i]['width'] = sizes[i][0]
      roidb[i]['height'] = sizes[i][1]
    # need gt_overlaps as a dense array for argmax
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)
    roidb[i]['max_classes'] = max_classes
    roidb[i]['max_overlaps'] = max_overlaps
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    nonzero_inds = np.where(max_overlaps > 0)[0]
    assert all(max_classes[nonzero_inds] != 0)


def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.    
    
    ratio_list = []
    for i in range(len(roidb)):
      width = roidb[i][0]['width']
      height = roidb[i][0]['height']
      ratio = width / float(height)

      if ratio > ratio_large:
        roidb[i][0]['need_crop'] = 1
        ratio = ratio_large
      elif ratio < ratio_small:
        roidb[i][0]['need_crop'] = 1
        ratio = ratio_small        
      else:
        roidb[i][0]['need_crop'] = 0

      ratio_list.append(ratio)
    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
      if len(roidb[i]['boxes']) == 0:
        del roidb[i]
        i -= 1
      i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb

def combined_roidb(imdb_names, training=True, duplicate_frames=False):
  """
  Combine multiple roidbs
  """

  def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
      print('Appending horizontally-flipped training examples...')
      imdb.append_flipped_images()
      print('done')

    print('Preparing training data...')

    prepare_roidb(imdb)
    print('done')

    return imdb.roidb
  
  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb
  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]

  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)

  if training:
    roidb = filter_roidb(roidb)
  roidb_pairs = create_roi_pairs(roidb, training, duplicate_frames)
  if training:
    np.random.seed(123)
    np.random.shuffle(roidb_pairs)
  ratio_list, ratio_index = rank_roidb_ratio(roidb_pairs)

  return imdb, roidb_pairs, ratio_list, ratio_index
