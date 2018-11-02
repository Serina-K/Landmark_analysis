#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:32:25 2018

@author: zeynep
"""
from os import listdir

import pickle
from pathlib import Path
import numpy as np

import constants
from importlib import reload
reload(constants)

import file_tools as file_tools
import clip_tools as clip_tools



class Clip_builder():
    """
    Clip is a collection of frames. 
    """
    def __init__(self, illustrate = True):
        
        self.exp_info = file_tools.readExpInfo(constants.EXP_INFO_FNAME)
        self.landmark_fnames = sorted([constants.LANDMARK_DIR + f for f in listdir(constants.LANDMARK_DIR) if '.dat.npy' in f])   
        self.accmeter_fnames = sorted([constants.ACCMETER_DIR + f for f in listdir(constants.ACCMETER_DIR) if 'acc.npy' in f]) # add
    
    def build_all(self):
        
        print('Building library of clips...')

        for i in range(0, len(self.exp_info['video_fname'])):
            clip = {\
                    'video_fname': [],\
                    'info': {},\
                    'frames':[],\
                  'nblinks': 0\
                   }
                    
            clip['video_fname'] = self.exp_info['video_fname'][i]
                     
            # loading and filling in frames
            # clip['info'] can be modificed if number of landmarks is not enough
            temp_landmarks, temp_accmeter = self.load_data_files(clip)
            
            if clip['info']['valid']:
                
                for frame_no in range(constants.NFRAMES):
                    
                    frame = {'frame_no': 0,\
                              't0': 0,\
                              'landmarks':  [],\
                              'eye_size': {'right': 0, 'left': 0, 'mean': 0},\
                              'eye_aspect_ratio': {'right': 0, 'left': 0, 'mean': 0},\
                              'ocular_breadth':  {'interocular': 0, 'biocular': 0},\
                              'pose_LM': {'yaw':0, 'pitch': 0, 'roll':0},\
                              'pose_ACC': {'yaw':0, 'pitch': 0, 'roll':0}}
                     
                    t0 = clip['info']['clip_t0_unix'] + frame_no / constants.FPS
                    
                    frame['frame_no'] = frame_no
                    frame['t0'] = t0
                    frame['landmarks'] = self.load_landmarks(temp_landmarks, frame_no)
                    frame['eye_size'] = self.get_eyesize( frame['landmarks'] )
                    frame['eye_aspect_ratio'] = self.get_eye_aspect_ratio( frame['landmarks'] )
                    frame['ocular_breadth'] = self.get_ocular_breadth( frame['landmarks'] )
                    frame['pose_LM'] = self.get_poses_LM( frame['landmarks'])
                    frame['pose_ACC'] = self.load_poses_ACC(frame['t0'], frame['frame_no'], temp_accmeter)
                    

                    clip['frames'].append(frame)
                    
                
                # save clip variable
                fpath = Path(constants.CLIP_VAR_DIR + 'clipvar_' + clip['video_fname'] )
                with open(str(fpath), 'wb') as f:
                    pickle.dump(clip, f, pickle.HIGHEST_PROTOCOL)

                        

        
    def load_data_files(self, clip):
        """
        exp_info involves details about the experiments. Currently the data 
        fields are:
            
        video_fname, participant, experiment, clip_t0_unix, t0_min, timebin, 
        playlist, valid, annotator1, annotator2,
        
        I may change the file and add new fields.
        """
        
        # build a similar dict variable in this class with the same keys as cvs
        ind = np.where(np.array(self.exp_info['video_fname']) == clip['video_fname'])
        for key in self.exp_info.keys():
            clip['info'][key] = self.exp_info[key][ind[0][0]]
        
        # other data files: landmarks and accelemeter data from EEG head band
        landmark_fname = constants.LANDMARK_DIR + clip['video_fname'] + '.dat.npy'
        accmeter_fname = constants.ACCMETER_DIR + clip['video_fname'] + '_acc.npy'
        
        temp_landmarks = np.load(landmark_fname)
        temp_accmeter =  np.load(accmeter_fname)
        
        # double-check validity
        # if some frames miss landmarks, skip this clip
        if (len(temp_landmarks) < constants.NLANDMARKS_PER_CLIP):
            print('Clip {} \tN_landmarks {} \tN_accmeter {}'.format(clip['video_fname'], len(temp_landmarks), len(temp_accmeter)))
            clip['info']['valid'] = 0
            
        return temp_landmarks, temp_accmeter

        
    def load_landmarks(self, temp_landmarks, frame_no):
        """
        Each clip has 300 frames and 68 landmarks per frame.
        temp_landmarks keeps the landmarks for the entire clip. Here I extract 
        the part that relates only this frame (68 points)
        """

        ind_0 = frame_no * constants.NLANDMARKS_PER_FRAME
        ind_f = ind_0 +  constants.NLANDMARKS_PER_FRAME
        landmarks = temp_landmarks[ind_0: ind_f, :]
        
        return landmarks
    
    def get_eyesize(self, landmarks):
        """
        This function computes the mean size of the eyes (mean of right and left)
        for all frames.
        This means that the eyes may be closed or blinking etc.
        """
        eye_size = {'right': 0,\
                    'left' : 0, \
                    'mean' : 0}
        
        xpix = landmarks[:][:,0]
        ypix = 720 - landmarks[:][:,1] # dlib assumes that the origin is upper left corner

        size_r = clip_tools.PolyArea(xpix[36:42], ypix[36:42])
        size_l = clip_tools.PolyArea(xpix[42:48], ypix[42:48])
        size_m = (size_r + size_l)*0.5
    
        eye_size['right'] = size_r
        eye_size['left'] = size_l
        eye_size['mean'] = size_m
        
        return eye_size
    
    def get_eye_aspect_ratio(self, landmarks):
        """
        This function computes eye aspect ratio. the method is explained in
        
            Real-Time Eye Blink Detection using Facial Landmarks
            Tereza Soukupova and Jan Cech
            CVWW 2016
            
        TODO
        Use the method of the aboive paper to detect blinks
        """
        eye_aspect_ratio = {'right': 0, 'letf': 0, 'mean': 0}
        
        eye_aspect_ratio['right'] = (np.linalg.norm(landmarks[37] - landmarks[41]) + \
                        np.linalg.norm(landmarks[38] - landmarks[40]))\
                        / 2 / np.linalg.norm(landmarks[36] - landmarks[39])   
 
        eye_aspect_ratio['left'] = (np.linalg.norm(landmarks[43] - landmarks[47]) + \
                    np.linalg.norm(landmarks[44] - landmarks[46]))\
                    / 2 / np.linalg.norm(landmarks[42] - landmarks[45])  
                    
        eye_aspect_ratio['mean'] =  0.5 * (eye_aspect_ratio['right'] + eye_aspect_ratio['left'])
        
        return eye_aspect_ratio
    
    def get_ocular_breadth(self, landmarks):
        """
        This function computes the mean size of the eyes (mean of right and left)
        for all frames.
        This means that the eyes may be closed or blinking etc.
        """
        
        ocular_breadth = {'interocular': 0,\
                          'biocular': 0}
        
        xpix = landmarks[:][:,0]
        ypix = 720 - landmarks[:][:,1] # dlib assumes that the origin is upper left corner
            
        p_36 = np.array(xpix[36],ypix[36])
        p_39 = np.array(xpix[39],ypix[39])    
        p_42 = np.array(xpix[42],ypix[42])
        p_45 = np.array(xpix[45],ypix[45])            
    
        ocular_breadth['interocular'] = np.linalg.norm(p_39 - p_42)
        ocular_breadth['biocular'] = np.linalg.norm(p_36 - p_45)
        
        return ocular_breadth
    
    def load_poses_ACC(self, t0, frame_no, temp_accmeter): # add
        """
        This function loads the head pose from the accelerometer data (ACC) of 
        the EGG head band.
        
        It takes the content of the ***_acc.npy, which is the accelerometer info
        for the entire clip; and crops the relevant part 
        It computes the mean and std of yaw, pitch, roll
        
        This way we have an estimate for all 3 values.
        The estimation of roll angle is also supposed to be more reliable.
        
        The three columns should be organized as:
            Roll, Pitch, Yaw
        
        TODO
        Build the input array such that the first column has the timestamp
        How are the accelerometer data cropped?
        check whether estimation is stable, if not apply moving average filter etc
        check polyserial correlation of roll, yaw, pitch with labels
        """
        pose_ACC = {'pitch': 0,\
                    'roll' : 0,\
                    'yaw'  : 0}
        
        t0 = t0 + frame_no / constants.FPS
        tf = t0 + 1/constants.FPS
        
        ind = np.where(np.logical_and(\
                                     temp_accmeter[:,0] > t0, \
                                     temp_accmeter[:,0] <= tf))[0].tolist()
      
        # Note that col 0 is time stamp, and cols 1, 2, 3 are the angles
        # TODO 
        # I am not sure which column is yaw, pitch roll (1, 2, 3)
        # check it
        if len(ind) > 0: 
            pose_ACC['pitch'] = np.mean(temp_accmeter[ind, 1]) * np.pi # + means down, - means up
            pose_ACC['roll'] = np.mean(temp_accmeter[ind, 2]) * np.pi 
            pose_ACC['yaw'] = np.mean(temp_accmeter[ind, 3])  * np.pi # + means looking left
            
        return pose_ACC
    
        
    def get_poses_LM(self, landmarks):
        """
        This function computes the head pose values from landmarks (LM)
        This it returns only roll.
        Pitch and yaw angles are returned as 0

        Basically it fits a line to the axis along the nose
        And computes roll angle as the slope of that line
        
        TODO 
        Currently I take points 27, 28, 29, 30
        The tip of the nose is 33 but it is not considered
        I can add that point too 
        """
            
        pose_LM = {'pitch': 0, \
                    'roll' : 0,\
                    'yaw'  : 0}
                
        nose_x = landmarks[27:31][:, 0]
        nose_y = 720 - landmarks[27:31][:, 1] # dlib assumes that the origin is upper left corner
        
        fit = np.polyfit(nose_x, nose_y, 1)
        fit_fn = np.poly1d(fit) # for display purposes, maybe reduntand sometimes
        
        temp_angle = np.arctan(fit[0]) +np.pi/2# between -pi and pi
        
        pose_LM['roll'] = clip_tools.limit_range(temp_angle)
        
        return pose_LM
