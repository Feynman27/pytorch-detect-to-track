import os
import glob
import shutil
import subprocess
import time
from collections import deque

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
    def __init__(self, pred_boxes, scores, pred_trk_boxes, classes, video_id=''):
        print("Starting post-processing on video id {}".format(video_id))
        self.video_id = video_id
        self.pred_boxes = pred_boxes
        self.scores = scores
        self.pred_trk_boxes = pred_trk_boxes
        self.num_frame_pairs = pred_boxes.size(0)
        self.num_frames = self.num_frame_pairs+1
        self.num_classes = len(classes)
        self.classes = classes
        self.class_agnostic = True
        self.jumpgap = 5
	self.alpha_l = 3.0 
        self.all_paths = np.ndarray(shape=(self.num_classes,), dtype=np.object)

    def class_paths(self, path_score_thresh=0.0):
        start_paths = time.time()
        # Iterate over all det categories and generate paths for each class
        self.generate_paths()
        # Perform temporal labeling
        final_tubes = self.get_tubes()
        end_paths = time.time()
        print('Tube generation done for all classes. Took {} sec'.format(end_paths-start_paths))
        keep = torch.nonzero(final_tubes['dpPathScore']>path_score_thresh).view(-1)
        self.path_total_score = final_tubes['path_total_score'][keep]
        self.path_scores = final_tubes['path_scores'][keep.cpu().numpy()]
        self.path_boxes = final_tubes['path_boxes'][keep.cpu().numpy()]
        self.path_starts = final_tubes['starts'][keep]
        self.path_ends = final_tubes['ends'][keep]
        self.path_labels = final_tubes['label'][keep]
        return final_tubes

    def dpEM_max(self, M):
        M = M.t() # classes X length of tubes[frames]
        r,c = M.size(0), M.size(1)
        D = torch.zeros(r,c+1).cuda() # add extra column
        D[:,1:] = M.clone()
        v = torch.arange(r).cuda()
        # phi stores all max indices in the forward pass
        phi = torch.zeros(r,c).cuda()
        # For each frame, get the max score for each class. The term alpha_l*(v!=i) will add 
        # a penalty by subtracting alpha_l from the data term for all classes other than the ith class.
        # The vector (v!=i) will return a logical tensor consisting of r elements. The ith location
        # is 0, and all other locations are 1. This way, all classes other than the ith class
        # are multiplied by alpha_l. For each ith iteraction (over classes), we get the max value and 
        # add it to the data term D[i,j]. This way, the max value for the ith class is stored in the 
        # jth frame.  
        for j in range(1,c+1): # frame index
            for i in range(r): # class index
                # Get max det score for frame j-1, penalize classes != ith class
                dmax,tb = torch.max(D[:,j-1]-self.alpha_l*(v!=i).float(), dim=0, keepdim=True)
                # For ith class, sum scores across frames
                D[i,j]+=dmax[0]
                # For ith class and j-1st frame, assign label with max score
                phi[i,j-1] = tb[0]
        # Traceback from last frame
        D = D[:, 1:]
        q = c
        # predicted class in last frame of tube
        _, p = torch.max(D[:,-1], dim=0, keepdim=True)
        i=p[0] # index of max element in last frame of D
        j=q-1 # frame indices
        p = deque([i+1]) # add 1 to i since class index 0 is __background__ class
        q = deque([j]) # jth frame in tube, start at end of tube
        while j>0: # quit when you hit the first frame in the tube
            tb = int(phi[i,j]) # i:index of max element in last frame of D, j:last frame index
            p.appendleft(tb+1)
            q.appendleft(j-1)
            j-=1
            i = tb
             
        return torch.FloatTensor(p).cuda(), torch.FloatTensor(q).cuda(), D
        
                

    def extract_action(self, _p, _q, _D, action):
        '''
        Extract frames in path where label=action
        '''
        inds = torch.nonzero(_p==action)
        if inds.numel()==0:
            ts = torch.FloatTensor([]).cuda()  
            te = torch.FloatTensor([]).cuda()  
            scores = torch.FloatTensor([]).cuda()  
            label = torch.FloatTensor([]).cuda()  
            total_score = torch.FloatTensor([]).cuda()  
        else:
            inds_diff = \
                torch.cat([inds, (inds[-1]+1).view(-1,1)],dim=0) - torch.cat([(inds[0]-2).view(-1,1),inds],dim=0)
            inds_diff = inds_diff.view(-1)
            ts = torch.nonzero(inds_diff>1)
            inds = inds.view(-1)
            if ts.size(0)>1: # 2 starting points for label=action
                te = torch.cat([ts.view(-1)[1:]-1, torch.cuda.LongTensor([inds.size(0)-1])])
            else:
                te = inds.size(0)-1
                te = torch.cuda.LongTensor([te]) 
            ts = torch.index_select(inds, 0, ts.view(-1))
            te = torch.index_select(inds, 0, te.view(-1))
            dt = te - ts
            q_s = torch.index_select(_q, 0, ts.view(-1)).long()
            q_e = torch.index_select(_q, 0, te.view(-1)).long()
            D_e = torch.index_select(_D, 1, q_e)[action]
            D_s = torch.index_select(_D, 1, q_s)[action]
            scores = ((D_e-D_s)/(dt.float()+1e-6)).view(-1,1)
            label = torch.cuda.FloatTensor(ts.size(0),1).fill_(1) * action
            total_score = torch.cuda.FloatTensor(ts.size(0),1).fill_(1) \
                              * _D[int(_p[-1]), int(_q[-1])]/ _p.size(0) 
            
            
        return ts,te,scores,label,total_score
        

    def get_tubes(self):
	'''
	Facade function for smoothing tubes.
	'''
	num_classes = self.num_classes
	counter=0
        final_tubes = {'starts':[], 'ends':[], 'ts':[], 'video_id':[], 
                       'te':[], 'dpActionScore':[], 'label':[],
                       'dpPathScore':[], 
                       'path_total_score':[], 'path_boxes':[], 'path_scores':[]}
       	# Perform temporal trimming
	for cls_ix in range(1,num_classes): # skip background class
            print('Performing temporal smoothing for class {}'.format(self.classes[cls_ix]))
	    # get paths for cls_ix
	    class_paths = self.all_paths[cls_ix]
            if class_paths is None: # skip classes with no paths
	        continue
            num_paths = len(self.all_paths[cls_ix]['count']) # num paths for cls_ix
            for i_pth in range(num_paths):
                M = class_paths['all_scores'][i_pth].clone()[:, 1:] # softmax across classes (exclude bkg)
                pred_path, time, D = self.dpEM_max(M)
                Ts,Te,Scores, Label, DpPathScore = self.extract_action(pred_path, time, D, cls_ix)
                if Ts.numel()==0:
                    continue
                for k in range(Ts.numel()):
                    final_tubes['starts'].append(class_paths['start'][i_pth])
                    final_tubes['ends'].append(class_paths['end'][i_pth])
                    final_tubes['ts'].append(Ts[k]) # where tube starts for this class
                    final_tubes['video_id'].append(self.video_id)
                    final_tubes['te'].append(Te[k]) # where tube end for this class
                    final_tubes['dpActionScore'].append(Scores[k]) 
                    final_tubes['label'].append(Label[k])
                    final_tubes['dpPathScore'].append(DpPathScore[k])
                    final_tubes['path_total_score'].append(class_paths['scores'][i_pth].mean())
                    final_tubes['path_boxes'].append(class_paths['boxes'][i_pth])
                    final_tubes['path_scores'].append(class_paths['scores'][i_pth])
        final_tubes['starts'] = torch.cat(final_tubes['starts'], dim=0)
        final_tubes['ends'] = torch.cat(final_tubes['ends'], dim=0)
        final_tubes['ts']=torch.cuda.LongTensor(final_tubes['ts'])
        final_tubes['te']=torch.cuda.LongTensor(final_tubes['te'])
        final_tubes['dpActionScore'] = torch.cat(final_tubes['dpActionScore'],dim=0)
        final_tubes['label']=torch.cat(final_tubes['label'],dim=0)
        final_tubes['dpPathScore']=torch.cat(final_tubes['dpPathScore'],dim=0)
        final_tubes['path_total_score']=torch.cuda.FloatTensor(final_tubes['path_total_score'])
        final_tubes['path_boxes']=np.array(final_tubes['path_boxes'], dtype=np.object)
        final_tubes['path_scores']=np.array(final_tubes['path_scores'], dtype=np.object)
        return final_tubes            

    def generate_paths(self):
        for cls_ix in range(1, self.num_classes): # skip background
            all_scores = np.ndarray(shape=(self.num_frame_pairs,), dtype=np.object)
	    cls_boxes = np.ndarray(shape=(self.num_frame_pairs,), dtype=np.object)
	    cls_scores = np.ndarray(shape=(self.num_frame_pairs,), dtype=np.object)
            print('Class: {}'.format(self.classes[cls_ix]))
            self._curr_class = self.classes[cls_ix]
            for pair_ix in range(self.num_frame_pairs):
                boxes_t0 = self.pred_boxes[pair_ix][0].clone()
                scores_t0 = self.scores[pair_ix][0][:,cls_ix].clone()
                pick = torch.nonzero(scores_t0>0.0).view(-1)
                # If no good scores for this frame/class, go to next frame
                assert pick.numel()>0, "No detections found for this class."
                if pick.numel()==0:
                    all_scores[pair_ix] = torch.cuda.FloatTensor(0) # empty tensor
                    cls_boxes[pair_ix] = torch.cuda.FloatTensor(0) # empty tensor
                    cls_scores[pair_ix] = torch.cuda.FloatTensor(0) # empty tensor
                    continue 
                # Get scores that passed filter and sort highest-->lowest
                scores_t0 = scores_t0[pick]
                boxes_t0 = boxes_t0[pick, :]
                all_scores_t0 = self.scores[pair_ix][0][pick, :]
                _, pick = torch.sort(scores_t0, descending=True)
                # Take at most 50 per frame per class
                to_pick = min(50,pick.numel())
                pick = pick[:to_pick]
                scores_t0 = scores_t0[pick]
                boxes_t0 = boxes_t0[pick,:]
                all_scores_t0 = all_scores_t0[pick,:]
                cls_dets_t0 = torch.cat([boxes_t0, scores_t0.contiguous().view(-1,1)], dim=1)
                pick = nms(cls_dets_t0, 0.3)
                # TODO check pick is sorted in descending order
                # Take top 10 dets after nms
                pick = pick.view(-1).long()                
                pick = pick[:min(10, pick.numel())]

                cls_boxes[pair_ix] = boxes_t0[pick, :].clone()
                cls_scores[pair_ix] = scores_t0[pick].clone()
                all_scores[pair_ix] = all_scores_t0[pick, :].clone()

            paths = self.incremental_linking(cls_boxes, cls_scores, all_scores)
            self.all_paths[cls_ix] = paths

    def get_path_count(self, live_paths_boxes):
        return len(live_paths_boxes)

    def bbox_overlaps(self, anchors, gt_boxes):
        """
        anchors: (N, 4) ndarray of float
        gt_boxes: (K, 4) ndarray of float

        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        N = anchors.size(0)
        K = gt_boxes.size(0)

        gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                    (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

        anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                    (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

        boxes = anchors.view(N, 1, 4).expand(N, K, 4)
        query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

        iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
            torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
            torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
        ih[ih < 0] = 0

        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        return overlaps

        
    def fill_gaps(self, paths_boxes, paths_scores, paths_all_score, paths_path_score, 
                        paths_found_at, paths_count, paths_last_found):
        '''
        paths: nd.objectarray of torch.Tensors
        gap: threshold for stale tube (in units of frames)
        '''
        gap = self.jumpgap
        gap_filled_paths_start = []
        gap_filled_paths_end = []
        gap_filled_paths_path_score = []
        gap_filled_paths_found_at = []
        gap_filled_paths_count = []
        gap_filled_paths_last_found = []
        gap_filled_paths_boxes = []
        gap_filled_paths_scores = []
        gap_filled_paths_all_scores = []

        g_count = 0
        path_count = self.get_path_count(paths_boxes)
        for lp in range(path_count):
            if paths_found_at[lp].size(0) > gap: # do we have at least @gap boxes in tube
                gap_filled_paths_start.append([])
                gap_filled_paths_end.append([])
                gap_filled_paths_path_score.append([])
                gap_filled_paths_found_at.append([])
                gap_filled_paths_count.append([])
                gap_filled_paths_last_found.append([])
                gap_filled_paths_boxes.append([])
                gap_filled_paths_scores.append([])
                gap_filled_paths_all_scores.append([])


                gap_filled_paths_start[g_count].append(paths_found_at[lp][0]) # start frame of tube
                gap_filled_paths_end[g_count].append(paths_found_at[lp][-1]) # end frame of tube
                gap_filled_paths_path_score[g_count].append(paths_path_score[lp].clone()) # path score
                gap_filled_paths_found_at[g_count].append(paths_found_at[lp].clone()) # trail of frames
                gap_filled_paths_count[g_count].append(paths_count[lp]) # boxes in tube
                gap_filled_paths_last_found[g_count].append(paths_last_found[lp]) # frames where last found
                count = 0
                i = 0 # index of box in tube
                while i < paths_scores[lp].size(0):
                    diff_found = (paths_found_at[lp][i]-paths_found_at[lp][max(0,i-1)])[0] 
                    if count == 0 or diff_found==1:
                        gap_filled_paths_boxes[g_count].append(paths_boxes[lp][i,:].clone().unsqueeze(0))
                        gap_filled_paths_scores[g_count].append(paths_scores[lp][i].clone().unsqueeze(0))
                        gap_filled_paths_all_scores[g_count].append(paths_all_score[lp][i,:]\
                                                                    .clone().unsqueeze(0))
                        i+=1
                        count+=1
                    else: # boxes in tube are > 1 frame apart, so fill the gap with the ith box
                        for d in range(diff_found):
                            gap_filled_paths_boxes[g_count].append(paths_boxes[lp][i,:].clone().unsqueeze(0))
                            gap_filled_paths_scores[g_count].append(paths_scores[lp][i].clone().unsqueeze(0))
                            gap_filled_paths_all_scores[g_count].append(paths_all_score[lp][i,:]\
                                                                        .clone().unsqueeze(0))
                            count+=1
                        i+=1
                g_count+=1  
                
        return gap_filled_paths_boxes, gap_filled_paths_scores, gap_filled_paths_all_scores, \
               gap_filled_paths_path_score, gap_filled_paths_found_at, \
               gap_filled_paths_count, gap_filled_paths_last_found, \
               gap_filled_paths_start, gap_filled_paths_end


    def incremental_linking(self, frames_boxes, frames_scores, frames_all_scores):
        # Online path building
        # dead-path count
        dp_count = 0
	for t0 in range(self.num_frame_pairs):
	    # boxes detected in frame t0
            if frames_boxes[t0].numel()==0:
                num_boxes = 0
            else:
                num_boxes = frames_boxes[t0].size(0)
            assert num_boxes>0, 'Must have boxes for class to build tubes. Check your filter threshold.'
            if t0==0: # If on first frame pair, initialize with all detections
                live_paths_boxes = np.ndarray(shape=(num_boxes, ), dtype=np.object)
                live_paths_scores = np.ndarray(shape=(num_boxes, ), dtype=np.object)
                live_paths_all_scores = np.ndarray(shape=(num_boxes,), dtype=np.object)
                live_paths_path_score = np.ndarray(shape=(num_boxes, ), dtype=np.object)
                live_paths_found_at = np.ndarray(shape=(num_boxes, ), dtype=np.object)
                live_paths_count = np.ndarray(shape=(num_boxes, ), dtype=np.object)
                live_paths_last_found = np.ndarray(shape=(num_boxes, ), dtype=np.object)

                for b in range(num_boxes):
                    live_paths_boxes[b] = frames_boxes[t0][b,:].clone().unsqueeze(0) 
                    live_paths_scores[b] = frames_scores[t0][torch.cuda.LongTensor([b])].clone().unsqueeze(0)
                    live_paths_all_scores[b] = frames_all_scores[t0][b].clone().unsqueeze(0)
                    live_paths_path_score[b] = frames_scores[t0][torch.cuda.LongTensor([b])].clone().unsqueeze(0)
                    live_paths_found_at[b] = torch.cuda.LongTensor([0]).unsqueeze(0)
                    live_paths_count[b] = 1
                    live_paths_last_found[b] = 0 # last time found from current frame
            else: # frames after the first
                lp_count = self.get_path_count(live_paths_boxes) # get live-path count
                print('Live paths in frame {} for class {}: {}'.format(t0, self._curr_class, lp_count))
                # last box in each live path
                last_boxes_lps = torch.cat([box[-1].unsqueeze(0) for box in live_paths_boxes], dim=0)
                # iou between boxes in last frame of tube and dets in current frame
                iou = self.bbox_overlaps(last_boxes_lps, frames_boxes[t0].clone())
                # Take scores in current frame dets that have iou above 0.1
                edge_scores = frames_scores[t0].clone().expand(lp_count, num_boxes)
                edge_scores = edge_scores*(iou>0.1).float()
                #edge_scores = torch.zeros(lp_count,num_boxes).cuda()
                dead_count = 0
                covered_boxes = torch.zeros(1,num_boxes).cuda()
                path_order_score = torch.zeros(1, lp_count).cuda()
                for lp in range(lp_count):
                    # Is the path live (has it been found in last jumpgap frames)?
                    if live_paths_last_found[lp] < self.jumpgap:
                        # scores of boxes in current frame t0 that overlap with lpth path
                        box_to_lp_score = edge_scores[lp, :].clone() 
                        if box_to_lp_score.sum()>0: # check if there's at least one box match
                            # get box with max score 
                            m_score, max_ind = box_to_lp_score.max(0)
                            # Add box/score to live path
                            live_paths_count[lp]+=1 # increment boxes in live path
                            live_paths_boxes[lp] = torch.cat([live_paths_boxes[lp], 
                                                              frames_boxes[t0][max_ind,:]], dim=0)
                            live_paths_scores[lp] = torch.cat([live_paths_scores[lp], 
                                                              frames_scores[t0][max_ind].view(1,-1)], dim=0)
                            live_paths_all_scores[lp] = torch.cat([live_paths_all_scores[lp], 
                                                              frames_all_scores[t0][max_ind]], dim=0)
                            live_paths_path_score[lp]+=m_score # running sum of box scores in this live path
                            # trail of frames boxes were found in
                            live_paths_found_at[lp] = torch.cat([live_paths_found_at[lp],
                                                                 torch.cuda.LongTensor([t0]).view(1,-1)], dim=0)
                            live_paths_last_found[lp] = 0 # refresh tube
                            edge_scores[:, max_ind] = 0.0 # remove box from available boxes 
                            covered_boxes[0][max_ind] = 1.0 # remove box from available boxes
                        else: # don't add det boxes to live path, but keep track of how many frames have passed
                            live_paths_last_found[lp] += 1 # tube is more stale 
                    
                        scores, _ = torch.sort(live_paths_scores[lp])
                        num_sc = scores.numel()
                        path_order_score[:, lp] = scores[max(0,num_sc-self.jumpgap):].mean()
                    else: # path is dead
                        dead_count+=1

            
                # Sort path based on box scores and terminate dead paths
                _, path_inds = torch.sort(path_order_score, descending=True)
                path_inds = path_inds.view(-1)
                sorted_live_paths_boxes = [] 
                sorted_live_paths_scores = [] 
                sorted_live_paths_all_scores = [] 
                sorted_live_paths_path_score = [] 
                sorted_live_paths_found_at = [] 
                sorted_live_paths_count = [] 
                sorted_live_paths_last_found = [] 

                dead_paths_boxes = [] 
                dead_paths_scores = [] 
                dead_paths_all_scores = [] 
                dead_paths_path_score = [] 
                dead_paths_found_at = [] 
                dead_paths_count = [] 
                dead_paths_last_found = [] 

                lpc=0
                for lp in range(lp_count):
                    olp = path_inds[lp]
                    if live_paths_last_found[lp] < self.jumpgap:
                        sorted_live_paths_boxes.append(live_paths_boxes[olp].clone())
                        sorted_live_paths_scores.append(live_paths_scores[olp].clone())
                        sorted_live_paths_all_scores.append(live_paths_all_scores[olp].clone())
                        sorted_live_paths_path_score.append(live_paths_path_score[olp].clone())
                        sorted_live_paths_found_at.append(live_paths_found_at[olp].clone())
                        sorted_live_paths_count.append(live_paths_count[olp])
                        sorted_live_paths_last_found.append(live_paths_last_found[olp])
                        
                        lpc += 1
                    else:
                        dead_paths_boxes.append(live_paths_boxes[olp].clone())
                        dead_paths_scores.append(live_paths_scores[olp].clone())
                        dead_paths_all_scores.append(live_paths_all_scores[olp].clone())
                        dead_paths_path_score.append(live_paths_path_score[olp].clone())
                        dead_paths_found_at.append(live_paths_found_at[olp].clone())
                        dead_paths_count.append(live_paths_count[olp])
                        dead_paths_last_found.append(live_paths_last_found[olp])
                        dp_count+=1
                lp_count = self.get_path_count(sorted_live_paths_scores) # update live-path count
                
                # Start new paths using unassigned boxes
                if covered_boxes.sum() < num_boxes:
                    for b in range(num_boxes):
                        if covered_boxes[0][b] == 0: # box is not covered and is available
                            lp_count+=1 # new live paths
                            sorted_live_paths_boxes.append(frames_boxes[t0][b,:].clone().unsqueeze(0))
                            sorted_live_paths_scores.append(frames_scores[t0][torch.cuda.LongTensor([b])]\
							    .clone().unsqueeze(0))
                            sorted_live_paths_all_scores.append(frames_all_scores[t0][b].clone().unsqueeze(0))
                            sorted_live_paths_path_score.append(frames_scores[t0][torch.cuda.LongTensor([b])]\
                                                            .clone().unsqueeze(0))
                            sorted_live_paths_found_at.append(torch.cuda.LongTensor([t0]).unsqueeze(0))
                            sorted_live_paths_count.append(1)
                            sorted_live_paths_last_found.append(0)
              
                # live paths/dead paths for next time step            
		live_paths_boxes = np.array(sorted_live_paths_boxes, dtype=np.object)
		live_paths_scores = np.array(sorted_live_paths_scores, dtype=np.object)
		live_paths_all_scores = np.array(sorted_live_paths_all_scores, dtype=np.object)
		live_paths_path_score = np.array(sorted_live_paths_path_score, dtype=np.object)
		live_paths_found_at = np.array(sorted_live_paths_found_at, dtype=np.object)
		live_paths_count = np.array(sorted_live_paths_count, dtype=np.object)
		live_paths_last_found = np.array(sorted_live_paths_last_found, dtype=np.object)

		dead_paths_boxes = np.array(dead_paths_boxes, dtype=np.object)
		dead_paths_scores = np.array(dead_paths_scores, dtype=np.object)
		dead_paths_all_scores = np.array(dead_paths_all_scores, dtype=np.object)
		dead_paths_path_score = np.array(dead_paths_path_score, dtype=np.object)
		dead_paths_found_at = np.array(dead_paths_found_at, dtype=np.object)
		dead_paths_count = np.array(dead_paths_count, dtype=np.object)
		dead_paths_last_found = np.array(dead_paths_last_found, dtype=np.object)


	live_paths = self.fill_gaps(live_paths_boxes, live_paths_scores, live_paths_all_scores, 
		       live_paths_path_score, live_paths_found_at, live_paths_count, 
		       live_paths_last_found)
	
	live_paths_boxes = live_paths[0]
	live_paths_scores = live_paths[1]
	live_paths_all_scores = live_paths[2]
	live_paths_path_score = live_paths[3]
	live_paths_found_at = live_paths[4]
	live_paths_count = live_paths[5]
	live_paths_last_found = live_paths[6]
	live_paths_start = live_paths[7]
	live_paths_end = live_paths[8]  

        # paths that died throughout the video, built from frame_start to frame_end
	dead_paths = self.fill_gaps(dead_paths_boxes, dead_paths_scores, dead_paths_all_scores, 
		       dead_paths_path_score, dead_paths_found_at, dead_paths_count, 
		       dead_paths_last_found)
	
	dead_paths_boxes = dead_paths[0]
	dead_paths_scores = dead_paths[1]
	dead_paths_all_scores = dead_paths[2]
	dead_paths_path_score = dead_paths[3]
	dead_paths_found_at = dead_paths[4]
	dead_paths_count = dead_paths[5]
	dead_paths_last_found = dead_paths[6]
	dead_paths_start = dead_paths[7]
	dead_paths_end = dead_paths[8]  

        # extend live paths with dead paths
        live_paths_start.extend(dead_paths_start)
        live_paths_end.extend(dead_paths_end)
        live_paths_boxes.extend(dead_paths_boxes)
        live_paths_scores.extend(dead_paths_scores)
        live_paths_all_scores.extend(dead_paths_all_scores)
        live_paths_path_score.extend(dead_paths_path_score)
        live_paths_found_at.extend(dead_paths_found_at)
        live_paths_count.extend(dead_paths_count)
        live_paths_last_found.extend(dead_paths_last_found)

        # sort paths
        
	lp_count = self.get_path_count(live_paths_scores)
        path_order_score = torch.zeros(lp_count).cuda()
        for lp in range(lp_count):
            live_paths_start[lp] = torch.cat(live_paths_start[lp], dim=0)
            live_paths_end[lp] = torch.cat(live_paths_end[lp], dim=0)
            live_paths_boxes[lp] = torch.cat(live_paths_boxes[lp], dim=0)
            live_paths_scores[lp] = torch.cat(live_paths_scores[lp], dim=0)
            live_paths_all_scores[lp] = torch.cat(live_paths_all_scores[lp], dim=0)
            live_paths_path_score[lp] = torch.cat(live_paths_path_score[lp], dim=0)
            live_paths_found_at[lp] = torch.cat(live_paths_found_at[lp], dim=0)
            #live_paths_count[lp] = torch.cat(live_paths_count[lp], dim=0)
            #live_paths_last_found[lp] = torch.cat(live_paths_last_found[lp], dim=0)

            scores,_ = torch.sort(live_paths_scores[lp].view(-1), descending=True)
            num_sc = scores.numel()
            path_order_score[lp] = scores[:min(20,num_sc)].mean() 
        _, inds = torch.sort(path_order_score, descending=True)
        
        sorted_live_paths = {'start': [], 'end': [], 
                             'boxes': [], 'scores': [], 'all_scores':[],
                             'path_score': [], 'found_at': [], 'count': [], 'last_found': []}
        for lp in range(lp_count):
            olp = inds[lp] 
            sorted_live_paths['start'].append(live_paths_start[olp])
            sorted_live_paths['end'].append(live_paths_end[olp])
            sorted_live_paths['boxes'].append(live_paths_boxes[olp])
            sorted_live_paths['scores'].append(live_paths_scores[olp])
            sorted_live_paths['all_scores'].append(live_paths_all_scores[olp])
            sorted_live_paths['path_score'].append(live_paths_path_score[olp])
            sorted_live_paths['found_at'].append(live_paths_found_at[olp])
            sorted_live_paths['count'].append(live_paths_count[olp])
            sorted_live_paths['last_found'].append(live_paths_last_found[olp])
      
        return sorted_live_paths
                        
#############################################

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

