import os
import glob
import shutil
import subprocess

import numpy as np
import torch
import torch.utils.data as data
import cv2
from model.utils.blob import prep_im_for_blob, im_list_to_blob
from model.utils.config import cfg
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_overlaps

import pdb


class VideoPostProcessor(object):
    def __init__(self, pred_boxes, scores, pred_trk_boxes, classes):
        print("Starting post-processing.")
        self.pred_boxes = pred_boxes
        self.scores = scores
        self.pred_trk_boxes = pred_trk_boxes
        self.num_frames = pred_boxes.size(0)+1
        self.num_classes = len(classes)
        self.classes = classes
        self.top_scores = torch.cuda.FloatTensor(self.num_classes,1).zero_() 
        #self._aboxes = [[[] for _ in xrange(self.num_frames)] for _ in xrange(self.num_classes)]
        self._aboxes = np.ndarray(shape=(self.num_classes, self.num_frames), dtype=np.object)
        #self._box_inds = [[[] for _ in xrange(self.num_frames)] for _ in xrange(self.num_classes)]
        self._box_inds = np.ndarray(shape=(self.num_classes, self.num_frames), dtype=np.object)
        #self._track_boxes = [[[] for _ in xrange(self.num_frames)] for _ in xrange(self.num_classes)]
        self._track_boxes = np.ndarray((self.num_classes,self.num_frames), dtype=np.object)
        #self._paths = [[] for _ in xrange(self.num_classes)]
        self._paths = np.ndarray((self.num_classes,), dtype=np.object)
        #self._ascores_track =  [[] for _ in xrange(self.num_frames)]  
        self._ascores_track = np.ndarray(shape=(self.num_frames,), dtype=np.object)
        #self._aboxes_track = [[[] for _ in xrange(2)] for _ in xrange(self.num_frames)]
        self._aboxes_track = np.ndarray(shape=(self.num_frames, 2), dtype=np.object)
        self.CONF_THRESH = torch.cuda.FloatTensor(self.num_classes)
        self.CONF_THRESH.fill_(-1*float('Inf'))
        self.max_per_image = 400 # keep at most max_per_image dets per class b/4 nms
        self.max_per_set = 160*self.num_frames
        self.class_agnostic = True

        self._process_frame_pairs()
        
        for j in range(1, self.num_classes):
            self._aboxes[j], self._box_inds[j], self.CONF_THRESH[j] = \
                    self._keep_top_k(self._aboxes[j], self._box_inds[j], 
                            self.num_frames, self.max_per_set, self.CONF_THRESH[j])

    def build_class_paths(self):
        # Build paths for each object category in the video 
        print("Generating tracks...")
        for c in range(1, self.num_classes):
            # Get frames 
            frameIdx = torch.arange(0,self.num_frames).cuda()
            frameBoxes = self._aboxes[c] # boxes in each frame for the cth class
            nonempty_frames = []
            for f,boxes in enumerate(frameBoxes):
                if boxes is not None and boxes.size(0) != 0:
                    nonempty_frames.append(f)
            # Tensor of non-empty frame numbers
            nonempty_frames = torch.cuda.LongTensor(nonempty_frames)
            if nonempty_frames.size(0)>0:
                frameBoxes = frameBoxes[nonempty_frames.cpu().numpy()]
                frameIdx = frameIdx[nonempty_frames].long()
                aboxes_track = self._aboxes_track[frameIdx.cpu().numpy(), :]
                ascores_track = self._ascores_track[frameIdx.cpu().numpy()]
                X_track = [aboxes_track, ascores_track, c]
                # path for class c
                self._paths[c] = self._make_tubes(frameBoxes, 25, False, X_track)
                num_paths_c = self._paths[c]['boxes'].size(0)
                for j_pth in range(num_paths_c):
                    for k_frame in range(frameIdx.size(0)-1):
                        f_idx = frameIdx[k_frame]
                        temp_box = torch.cat([self._paths[c]['boxes'][j_pth][k_frame][:4].unsqueeze(0),  
                            torch.cuda.FloatTensor([self._paths[c]['scores'][j_pth][k_frame]]).unsqueeze(0) ], 
                            dim=1)
                        if self._track_boxes[c][f_idx] is None:
                            self._track_boxes[c][f_idx] = temp_box
                        else:
                            self._track_boxes[c][f_idx] = torch.cat([self._track_boxes[c][f_idx], temp_box], dim=0)
        print("Done!")
        return self._paths

    def _make_tubes(self, frameBoxes, max_per_image, box_voting, tracks_cell):
        '''Build tubes for cth class.
        '''

        tracks = {'boxes': None, 'scores': None, 'c': None}
        tracks['boxes'] = tracks_cell[0]
        tracks['scores'] = tracks_cell[1]
        tracks['c'] = tracks_cell[2]

        nms_thresh = 0.3

        object_frames_boxes = np.ndarray((len(frameBoxes),), dtype=np.object)
        object_frames_scores =  np.ndarray((len(frameBoxes),), dtype=np.object)
        object_frames_boxes_idx = np.ndarray((len(frameBoxes),), dtype=np.object)
        object_frames_trackedboxes = np.ndarray((len(frameBoxes),), dtype=np.object)
        
        # Iterate over the non-empty frames
        for f in range(len(frameBoxes)-1):
            # boxes in frame f
            boxes = frameBoxes[f]
            if box_voting: # TODO
                raise NotImplementedError
            else:
                nms_idx = nms(boxes[:, :5].clone(), nms_thresh).long().view(-1)
                if nms_idx.numel() > max_per_image:
                    nms_idx = nms_idx[:max_per_image]
                boxes = boxes[nms_idx]
                object_frames_boxes[f] = boxes[:, :4]
                object_frames_scores[f] = boxes[:, 4]
                object_frames_boxes_idx[f] = torch.arange(boxes.size(0)).cuda()
                if tracks['boxes'] is not None and tracks['boxes'][f, 0] is not None:
                    object_frames_trackedboxes[f] = tracks['boxes'][f, :]

        paths = self._zero_jump_link(object_frames_boxes, object_frames_scores,
                                     object_frames_boxes_idx, object_frames_trackedboxes)

        return paths


    def _zero_jump_link(self, frames_boxes, frames_scores, frames_boxes_idx, frames_trackedboxes):
        '''
         ---------------------------------------------------------
         Copyright (c) 2015, Georgia Gkioxari

         This file is part of the Action Tubes code and is available
         under the terms of the Simplified BSD License provided in
         LICENSE. Please retain this notice and LICENSE if you use
         this file (or any portion of it) in your project.

         At each iteration in the while loop, a new path is generated and the boxes, scores and box boxes_idx
         associated with this path are removed so that when the next path is generated, 
	 it will consider the remaining
         boxes. e.g. from frame 1 , out of 5 boxes, box no 2 is used to generate path 1, so we
         remove the box from the list, so when we will generate path 2, we will not use this box 2 again


         This version is corrected to keep track of the correct boxes indices in each
         path , as the boxes are removed after generating each path , to keep track of the indices
         of the boxes following lines are added :

         index = frames(V(j)).boxes_idx(id);
         index =  [index;  frames(V(j+1)).boxes_idx(id)]; %% newly added
         frames(V(j)).boxes_idx(id)  = []; %% newly added
         ---------------------------------------------------------
        :param frames_boxes:
        :param frames_scores:
        :param frames_boxes_idx:
        :param frames_trackedboxes:
        :return:
        '''
	
        # number of vertices
        num_frames = frames_boxes.shape[0]
        V = torch.arange(num_frames).long().cuda()
        # 0: not empty, 1: empty
        isempty_vertex = torch.cuda.LongTensor(V.size(0), 1).zero_()

        # Check: make sure data is not empty 
        # (note: last box is None since tracking has one less frame than detection)
        for i_bx in range(len(frames_boxes)-1):
            if frames_boxes[i_bx] is None or frames_boxes[i_bx].shape[0]==0:
                print('ERROR: Found empty box at {}'.format(i_bx))
                raise RuntimeError

	#--- dynamic programming (K-paths)
        path_counter = 0
        paths_total_score = []
        paths_idx = []
        paths_boxes = []
        paths_smooth_scores = []
        paths_scores = []

	T = len(V)-1
	# Break once we have an empty vertex (i.e. we've used all boxes in a frame)
	while not (isempty_vertex!=0).any():
	    data_scores = np.ndarray(shape=(T,), dtype=np.object)
            data_index = np.ndarray(shape=(T,), dtype=np.object)
            data_track_scores = np.ndarray(shape=(T,), dtype=np.object)
	    # Step 1: initialize data structures
	    for i in range(T):
	        num_states = frames_boxes[i].size(0) # num boxes in frame i
                data_scores[i] = torch.cuda.FloatTensor(num_states,1).zero_()
                data_index[i] = torch.cuda.FloatTensor(num_states,1).fill_(np.nan)
                data_track_scores[i] = torch.cuda.FloatTensor(num_states,1).fill_(np.nan)

            
            # Step 2: Solve Viterbi -- forward pass
            # Do a forward pass from the frame before the last frame in the video (frame T-1).
            # Compute the edge score between boxes at frame T to frame T-1. Continue this until
            # we reach the first frame. At the first iteration (T, T-1), compute the edge scores between
            # frames T and T-1 and add it to data_scores[i+1]. At the first iteration, all the scores in
            # data_scores[i+1] are zero. At the end of the first iteration, data_scores[i+1] is assigned the
            # max edge score for each box in the frame T-1. In the subsequent iteration, 
	    # we add the max edge score of
            # boxes from frame i+1 to the max edge scores of boxes in frame i. 
            # We then compute the max edge scores for
            # the ith frame and store these scores in data_scores[i]. 
            # The box indices from the boxes with max edge scores
            # with boxes from frame i+1 are stored in data_index[i].
            for i in range(T-2, -1, -1):
                frames_t1 = {'boxes':frames_boxes[i+1],
                             'scores':frames_scores[i+1],
                             'boxes_idx': frames_boxes_idx[i+1],
                             'trackedboxes': frames_trackedboxes[i+1]}
                frames_t0 = {'boxes':frames_boxes[i],
                             'scores':frames_scores[i],
                             'boxes_idx': frames_boxes_idx[i],
                             'trackedboxes': frames_trackedboxes[i]}
                edge_score, data_track_scores[i] = self._score_of_edge(frames_t0, frames_t1)
                edge_score = edge_score + data_scores[i+1].t()
                data_scores[i], data_index[i] = torch.max(edge_score, dim=1)
                data_scores[i] = data_scores[i].contiguous().view(-1,1)
            # Step 3: Decode -- backward pass of Viterbi -- backtracing
            path_counter += 1
            # sort max edge scores of boxes in first frame
            s,si = torch.sort(data_scores[0],descending=True,dim=0)
            id = si[0] # index of box having max edge score in first frame
            score = data_scores[0][id] # get score of box with id
            index = frames_boxes_idx[V[0]][id] # get id of box from frame V[0]
            boxes = frames_boxes[V[0]][id, :] # get box coordinates from top box at frame V[0]
            scores = frames_scores[V[0]][id] # get box score of top box from frame V[0]
            for j in range(T-1):
                id = data_index[j][id] # get the idth box from frame j
                index = torch.cat([index, frames_boxes_idx[V[j+1]][id]])
                boxes = torch.cat([boxes, frames_boxes[V[j+1]][id]])
                scores = torch.cat([scores, frames_scores[V[j+1]][id]])
            paths_total_score.append(score/num_frames)
            paths_idx.append(index)
            paths_boxes.append(torch.cat([boxes, scores.view(-1,1)], dim=1))
            # sort max edge scores of boxes in decreasing order
            top_scores,_ = torch.sort(scores, descending=True)
            #top_50 = torch.ceil(torch.FloatTensor([0.5*len(top_scores)])).long().cuda()
            mean_top_scrs = torch.mean(top_scores[:int(np.ceil(0.5*len(top_scores)))])
            # Use approx. Gaussian filter on scores
            gfilter = np.array([1.0,4.0,6.0,4.0,1.0])/16.0
            #smooth_scores = cv2.filter2D(scores.cpu().numpy(), -1, gfilter)
            smooth_scores = torch.from_numpy(cv2.filter2D(scores.cpu().numpy(), -1, gfilter)).cuda()
            smooth_scores += mean_top_scrs
            paths_smooth_scores.append(smooth_scores)
            scores += mean_top_scrs
            paths_scores.append(scores)

            # Step 4: Remove covered boxes
            for j in range(T):
                id = paths_idx[path_counter-1][j]
                # remove box id since it's been included in the path
                #id2Rem = torch.nonzero(frames_boxes_idx[V[j]]==id)
                ids2Keep = torch.nonzero(frames_boxes_idx[V[j]]!=id).view(-1)
                isempty_vertex[j] = int(ids2Keep.numel()==0)
                if ids2Keep.numel()==0:
                    continue
                frames_boxes[V[j]] = frames_boxes[V[j]][ids2Keep]
                frames_scores[V[j]] = frames_scores[V[j]][ids2Keep]
                frames_boxes_idx[V[j]] = frames_boxes_idx[V[j]][ids2Keep]

	paths = {'total_score': torch.stack(paths_total_score),
		    'boxes': torch.stack(paths_boxes),
		    'idx': torch.stack(paths_idx),
		    'smooth_scores': torch.stack(paths_smooth_scores),
		    'scores': torch.stack(paths_scores)}
        return paths 


    
    def _score_of_edge(self, v1, v2):
        N1 = v1['boxes'].size(0)
        N2 = v2['boxes'].size(0)
        score = torch.cuda.FloatTensor(N1,N2).fill_(np.nan)
        track_score = torch.cuda.FloatTensor(N1,N2).fill_(np.nan)

        for i1 in range(N1):
            # scores of i1 box in frame i with all boxes in frame i+1
            scores2 = v2['scores'].contiguous().view(-1,1)
            scores1 = v1['scores'][i1]
            score[i1, :] = scores1 + scores2.t()

        if v1['trackedboxes'] is not None and v2['trackedboxes'] is not None:
            # overlaps between the boxes with tracked_boxes
            # overlaps (N1, N2)
            overlap_ratio_1 = bbox_overlaps(v1['boxes'].contiguous(), v1['trackedboxes'][0])
            overlap_ratio_2 = bbox_overlaps(v2['boxes'].contiguous(), v1['trackedboxes'][1])
            track_score = torch.mm(torch.round(overlap_ratio_1), torch.round(overlap_ratio_2).t())
            score[track_score>0.]+=1.0
            track_score = (track_score>0.).float()
        else:
            track_score = torch.cuda.FloatTensor(N1,N2).zero_()
        return score, track_score



    def _keep_top_k(self, boxes, box_inds, end_at, top_k, thresh):
        '''Set dynamic class threshold based on average detections per class per frame
        (before nms) and keep boxes above this threshold.
        '''
        box_list = boxes[:end_at].tolist()
        box_list = [box for box in box_list if box is not None]
        if len(box_list)==0:
            return
        X = torch.cat(box_list, dim=0)
        if X.size(0) == 0:
            return

        scores,_ = torch.sort(X[:,4], descending=True)
        # set threshold for this class
        thresh = scores[min(scores.numel(), top_k)]
        for image_index in range(end_at):
            if boxes[image_index] is not None and boxes[image_index].size(0)>0:
                bbox = boxes[image_index]
                keep = torch.nonzero(bbox[:,4]>=thresh).view(-1)
                if keep.numel()==0:
                    continue
                boxes[image_index] = bbox[keep]
                box_inds[image_index] = box_inds[image_index][keep]
        return boxes, box_inds, thresh

    def _process_frame_pairs(self):
        # iterate over frame pairs
        for i_pair in range(self.pred_boxes.size(0)-1):
            print('Post-processing detections from t={} t+tau={}'.format(i_pair, i_pair+1))
            # Get scores in first frame of pair above 0.01 (ignore bg score)
            tracklet_score, tracklet_cls = torch.max(self.scores[i_pair,0][:,1:], dim=1)
            tracklets = torch.nonzero(tracklet_score>0.01).view(-1)
            if tracklets.numel()>0:
                if self._ascores_track[i_pair] is None:
                    self._ascores_track[i_pair] = self.scores[i_pair,0][tracklets][:, 1:]
                else:
                    self._ascores_track[i_pair] = torch.cat([self._ascores_track[i_pair], 
                            self.scores[i_pair,0][tracklets][:, 1:]])
                #if i_pair==0: #forward track
                #    self._aboxes_track[i_pair, 0] = self.pred_boxes[i_pair,0][tracklets]
                #    self._aboxes_track[i_pair, 1] = self.pred_trk_boxes[i_pair][tracklets]
                #else: # backward track
                # Set boxes in first frame of pair to predicted boxes 
                # and set boxes in second frame in pair to predicted tracking boxes
                # TODO check this step. Original implementation seems to differ.
                if self._aboxes_track[i_pair, 0] is None:
                    self._aboxes_track[i_pair, 0] = self.pred_boxes[i_pair,0][tracklets]
                    self._aboxes_track[i_pair, 1] = self.pred_trk_boxes[i_pair][tracklets]
                else:
                    self._aboxes_track[i_pair, 0] = torch.cat([self._aboxes_track[i_pair, 0], \
                            self.pred_boxes[i_pair,0][tracklets]])
                    self._aboxes_track[i_pair, 1] = torch.cat([self._aboxes_track[i_pair, 1], 
                            self.pred_trk_boxes[i_pair][tracklets]])
            # For each leg in siamese net, resort frame boxes
            for i_leg in range(self.pred_boxes.size(1)):
                # global frame index in the video
                frameindex = i_pair + i_leg
                # predicted boxes for this leg of the frame pair
                boxes = self.pred_boxes[i_pair, i_leg]
                scores = self.scores[i_pair, i_leg]
                # retrieve class predictions
                for i_cls in range(1,self.num_classes): # skip background class
                    inds = torch.nonzero(scores[:, i_cls] > self.CONF_THRESH[i_cls]).view(-1)
                    if inds.numel()>0:
                        _, ord = torch.sort(scores[inds][:, i_cls], descending=True)
                        ord = ord[:min(ord.numel(), self.max_per_image)]
                        inds = inds[ord].view(-1)
                        cls_boxes = boxes[inds]
                        if self.class_agnostic:
                            cls_boxes = cls_boxes[:,:4]
                        else:
                            cls_boxes = cls_boxes[:, int(4*i_cls):int(4*i_cls+4)]
                        cls_scores = scores[inds][:, i_cls]
                        bg_scores = scores[inds][:, 0]
                        cls_entry = torch.cat([cls_boxes, cls_scores.unsqueeze(1), bg_scores.unsqueeze(1)], dim=1)

                        if self._aboxes[i_cls, frameindex] is None:
                            self._aboxes[i_cls, frameindex] = cls_entry
                        else:
                            self._aboxes[i_cls, frameindex] = \
                                    torch.cat([self._aboxes[i_cls, frameindex], cls_entry], dim=0)
                        
                        if self._box_inds[i_cls, frameindex] is None:
                            self._box_inds[i_cls, frameindex] = inds
                        else:
                            inds = inds + len(self._box_inds[i_cls, frameindex])
                            self._box_inds[i_cls, frameindex] = \
                                    torch.cat([self._box_inds[i_cls, frameindex], inds])
                    else:
                        # leave element (i_cls, frameindex) empty
                        continue
        return





class VideoDataset(data.Dataset):
    def __init__(self, video_list, det_classes):
        self.det_classes = det_classes
        self.num_classes = len(det_classes)
        self.n_videos = len(video_list)
        self.video_paths = video_list

        # Keep at most max_per_image dets per class per image before NMS
        self.max_per_image = 400
        # Number of legs in the siamese network
        self.n_legs = 2

    def __getitem__(self, idx):
        self._video_idx = idx
        self._video_blob = []

        self.video_name = self.video_paths[idx]
        # Initialize frame index of the current video
        self._frame_idx = 0
        self._extract_frames(self.video_paths[idx])
        self._create_video_blob()
        return self._video_blob

    def __len__(self):
        return self.n_videos

    def _create_video_blob(self):
        for i_frame in range(self._n_frames-1):
            _sample_blob = {}
            print("Video name: {} {}/{}".format(self.video_name, i_frame, self._n_frames-1))
            frame_data_t0 = cv2.imread(self._frame_paths[i_frame])
            frame_data_t1 = cv2.imread(self._frame_paths[i_frame+1])
            frame_blob_t0 = self._get_image_blob(frame_data_t0, i_frame)
            frame_blob_t1 = self._get_image_blob(frame_data_t1, i_frame+1)
            pt_frame_tensor_t0 = torch.from_numpy(frame_blob_t0['data']).cuda()
            pt_frame_tensor_t1 = torch.from_numpy(frame_blob_t1['data']).cuda()
            # Permute to (B,C,H,W)
            pt_frame_tensor_t0 = pt_frame_tensor_t0.permute(0,3,1,2).contiguous()
            pt_frame_tensor_t1 = pt_frame_tensor_t1.permute(0,3,1,2).contiguous()
            pt_info_tensor_t0 = torch.from_numpy(frame_blob_t0['im_info']).cuda()
            pt_info_tensor_t1 = torch.from_numpy(frame_blob_t1['im_info']).cuda()
            pt_frame_number_tensor_t0 = torch.from_numpy(frame_blob_t0['frame_number']).cuda().unsqueeze(0)
            pt_frame_number_tensor_t1 = torch.from_numpy(frame_blob_t1['frame_number']).cuda().unsqueeze(0)
            #_sample_blob['data'] = [frame_blob_t0['data'],frame_blob_t1['data']]
            _sample_blob['data'] = torch.cat([pt_frame_tensor_t0, pt_frame_tensor_t1], dim=0)
            #_sample_blob['im_info'] = [frame_blob_t0['im_info'],frame_blob_t1['im_info']]
            _sample_blob['im_info'] = torch.cat([pt_info_tensor_t0, pt_info_tensor_t1], dim=0)
            #_sample_blob['frame_number'] = [frame_blob_t0['frame_number'],frame_blob_t1['frame_number']]
            _sample_blob['frame_number'] = torch.cat([pt_frame_number_tensor_t0, pt_frame_number_tensor_t1], dim=0)
            self._video_blob.append(_sample_blob)

    def _extract_frames(self, v_path):
        '''Extract all fromes from @v_path

        :param v_path: full path to video
        :return: list of full paths to extracted video frames
        '''
        print("Extracting frames from {}".format(v_path))
        # Store frames in tmp dir by default
        tmp_dir = os.path.join('/tmp', os.path.basename(v_path))
        # If path exists, delete it
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)
        # create directory to dump output to
        save_dir = tmp_dir.replace('.mp4', '') + "_processed"
        # clear contents of output directory before saving
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        print("Saving to {}".format(save_dir))
        
        self._input_dir = tmp_dir
        self._output_dir = save_dir
        # TODO Make fps configurable at command line
        cmd = "ffmpeg -i %s -vf fps=10 %s" % (v_path,os.path.join(tmp_dir,'%09d.png'))
        # execute ffmpeg cmd
        subprocess.call(cmd,shell=True)
        # set frame paths of the current video
        self._frame_paths = sorted(glob.glob("%s/*.png" % tmp_dir))
        self._n_frames = len(self._frame_paths)
        self._max_per_set = 160*self._n_frames # average 160 dets per class per frame before nms
        print("Found {} frames".format(self._n_frames))
        return 

    def _get_image_blob(self, im, frame_id):
        '''Convert image into network input.
        :param im: BGR nd.array
        :param frame_id: frame number in the given video
        :return image (frame) blob
        '''
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
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        blob = im_list_to_blob(processed_ims)
        scales = np.array(im_scale_factors)

        blobs = {'data': blob}
        blobs['im_info'] = np.array(
                [[blob.shape[1], blob.shape[2], scales[0]]], dtype=np.float32)
        blobs['frame_number'] = np.array([[frame_id]])

        return blobs

