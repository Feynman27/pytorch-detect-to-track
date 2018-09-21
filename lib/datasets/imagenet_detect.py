import os
import pdb
from datasets.imdb import imdb
import numpy as np
import scipy.sparse
import cPickle
import uuid
from datasets.vid_eval import vid_eval
from datasets.imagenet_vid_eval_motion import vid_eval_motion
from model.utils.config import cfg

class imagenet_detect(imdb):
    ##
    # @brief To initialize the dataset reader
    #
    # @param image_set One of [train,val,trainval,test]
    # @param devkit_path Full path to the directory where the data created by
    # imagenet_datasets.py is stored
    #
    def __init__(self, image_set, devkit_path, det_or_vid):
        imdb.__init__(self, "imagenet_" + det_or_vid.lower() + image_set)
	self._det_vid = det_or_vid 
        self._root_path = devkit_path
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = devkit_path # Currently its same as the devkit_path
        
        self._classes = ('__background__',  # always index 0
                        'airplane', 'antelope', 'bear', 'bicycle',
                        'bird', 'bus', 'car', 'cattle',
                        'dog', 'domestic_cat', 'elephant', 'fox',
                        'giant_panda', 'hamster', 'horse', 'lion',
                        'lizard', 'monkey', 'motorcycle', 'rabbit',
                        'red_panda', 'sheep', 'snake', 'squirrel',
                        'tiger', 'train', 'turtle', 'watercraft',
                        'whale', 'zebra')
        

        self._classes_map = ('__background__',  # always index 0
                        'n02691156', 'n02419796', 'n02131653', 'n02834778',
                        'n01503061', 'n02924116', 'n02958343', 'n02402425',
                        'n02084071', 'n02121808', 'n02503517', 'n02118333',
                        'n02510455', 'n02342885', 'n02374451', 'n02129165',
                        'n01674464', 'n02484322', 'n03790512', 'n02324045',
                        'n02509815', 'n02411705', 'n01726692', 'n02355227',
                        'n02129604', 'n04468005', 'n01662784', 'n04530566',
                        'n02062744', 'n02391049')
        
        print("Number of classes: {}".format(self.num_classes))
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.JPEG'
        self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())

        # Dataset specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'top_k'       : 2000,
                       'use_diff'    : False,
                       'rpn_file'    : None}

        assert os.path.exists(self._devkit_path), \
                'imagenet devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Data', self._det_vid, self._image_set,
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self._data_path, 'ImageSets',
                self._det_vid, self._image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            lines = [x.strip().split(' ') for x in f.readlines()]
        if len(lines[0]) == 2:
            self._image_index = [x[0] for x in lines]
            self._frame_id = [int(x[1]) for x in lines]
            self._frame_len = [-1 for x in lines] # temp hack 
        else:
            self._image_index = ['%s' % x[0] for x in lines]
            #self._pattern = [x[0]+'/%06d' for x in lines]
            self._start_frame_id = [int(x[1]) for x in lines]
            self._frame_id = [int(x[2]) for x in lines]
            self._frame_len = [int(x[3]) for x in lines]
        # return image_set_index, frame_id
    
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        gt_roidb = [self._load_vid_annotation(idx,index)
                    for idx, index in enumerate(self.image_index)]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_vid_annotation(self, idx, index):

        """ given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped', 'f]
        """
        import xml.etree.ElementTree as ET
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)
        roi_rec['frame_id'] = self._frame_id[idx] # eg) frame id -> 000076
        #if hasattr(self,'frame_id'):
        index_dir = index.split('/')
        if len(index_dir)<3:
            roi_rec['video_snippet'] = index_dir[0]
        else:
            roi_rec['video_snippet'] = index_dir[1]
        #roi_rec['frame__id'] = self.frame_seg_id[iindex]
        roi_rec['frame_snippet_len'] = self._frame_len[idx]

        if self._det_vid == 'DET':
            filename = os.path.join(self._data_path, 'Annotations', 'DET', self._image_set, index + '.xml')
        else:
            filename = os.path.join(self._data_path, 'Annotations', 'VID', self._image_set, index + '.xml')

        tree = ET.parse(filename)
        size = tree.find('size')
        roi_rec['height'] = float(size.find('height').text)
        roi_rec['width'] = float(size.find('width').text)
        #im_size = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION).shape
        #assert im_size[0] == roi_rec['height'] and im_size[1] == roi_rec['width']

        #filename = tree.find('filename').text
        #if filename == 'ILSVRC2014_train_00057748':
        #    pdb.set_trace()
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros(shape=(num_objs, self.num_classes), dtype=np.float32)
        valid_objs = np.zeros((num_objs), dtype=np.bool)
        track_id = np.zeros((num_objs), dtype=np.uint16)
        class_to_index = dict(zip(self._classes_map, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = np.maximum(float(bbox.find('xmin').text), 0)
            y1 = np.maximum(float(bbox.find('ymin').text), 0)
            x2 = np.minimum(float(bbox.find('xmax').text), roi_rec['width']-1)
            y2 = np.minimum(float(bbox.find('ymax').text), roi_rec['height']-1)
            if not class_to_index.has_key(obj.find('name').text):
                continue
            valid_objs[ix] = True
            cls = class_to_index[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            if self._det_vid == 'DET':
                track_id[ix] = int(ix) # assign obj index
            else:
                track_id[ix] = int(obj.find('trackid').text)

        boxes = boxes[valid_objs, :]
        gt_classes = gt_classes[valid_objs]
        track_id = track_id[valid_objs]
        overlaps = overlaps[valid_objs, :]
        overlaps = scipy.sparse.csr_matrix(overlaps)

        assert (boxes[:, 2] >= boxes[:, 0]).all() 

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False,
                        'track_id': track_id})
        return roi_rec

    def _get_imagenetVid_results_file_template(self):
        # devkit/results/det_test_aeroplane.txt
        filename = 'det_' + self._image_set + '_{:s}.txt'
        base_path = os.path.join(self._devkit_path, 'results')
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _write_imagenetVid_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} Imagenet vid results file'.format(cls)
            filename = self._get_imagenetVid_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'Annotations','VID',self._image_set,
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'ImageSets','VID',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_imagenetVid_results_file_template().format(cls)
            rec, prec, ap = vid_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, roidb, output_dir):
        self._roidb = roidb 
        self._image_index = ['/'.join(roi_entry[0]['image'].split('/')[-3:])\
                                .replace('.JPEG','').replace('.jpeg', '')\
                                .replace('.jpg','').replace('.JPG','') \
                                for roi_entry in self._roidb]
        self._write_imagenetVid_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_imagenetVid_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    pass
