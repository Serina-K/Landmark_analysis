#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:32:25 2018

@author: zeynep
"""
import numpy as np

import pickle
from pathlib import Path
from os import listdir

from importlib import reload

import constants
reload(constants)

import file_tools as file_tools
import clip_tools as clip_tools


"""
Clip is a collection of frames. 
"""
def init_mats( illustrate = True):
    
    exp_info = file_tools.readExpInfo(constants.EXP_INFO_FNAME)
    landmark_fnames = sorted([constants.LANDMARK_DIR + f for f in listdir(constants.LANDMARK_DIR) if '.dat.npy' in f])   
    accmeter_fnames = sorted([constants.ACCMETER_DIR + f for f in listdir(constants.ACCMETER_DIR) if 'acc.npy' in f]) # add

    return exp_info, landmark_fnames, accmeter_fnames

def build_all():
    
    print('Building library of clips...')
    exp_info, landmark_fnames, accmeter_fnames = init_mats()

    for i in range(0, len(exp_info['video_fname'])):
        clip = {\
                'video_fname': [],\
                'info': {},\
                'frames':[],\
              'nblinks': 0\
               }
                
        clip['video_fname'] = exp_info['video_fname'][i]
                 
        # loading and filling in frames
        # clip['info'] can be modificed if number of landmarks is not enough
        temp_landmarks, temp_accmeter = load_data_files(clip, exp_info)
        
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
                frame['landmarks'] = load_landmarks(temp_landmarks, frame_no)
                frame['eye_size'] = get_eyesize( frame['landmarks'] )
                frame['eye_aspect_ratio'] = get_eye_aspect_ratio( frame['landmarks'] )
               
                frame['ocular_breadth'] = get_ocular_breadth( frame['landmarks'] )
                frame['eye_size_normalized'] = get_eyesize_normalized( frame )
                frame['pose_LM'] = get_poses_LM( frame['landmarks'])
                frame['pose_ACC'] = load_poses_ACC(frame['t0'], frame['frame_no'], temp_accmeter)
                

                clip['frames'].append(frame)
                
            
            # save clip variable
            fpath = Path(constants.CLIP_VAR_DIR + 'clipvar_' + clip['video_fname'] )
            with open(str(fpath), 'wb') as f:
                pickle.dump(clip, f, pickle.HIGHEST_PROTOCOL)

                    

    
def load_data_files(clip, exp_info):
    """
    exp_info involves details about the experiments. Currently the data 
    fields are:
        
    video_fname, participant, experiment, clip_t0_unix, t0_min, timebin, 
    playlist, valid, annotator1, annotator2,
    
    I may change the file and add new fields.
    """
    
    # build a similar dict variable in this class with the same keys as cvs
    ind = np.where(np.array(exp_info['video_fname']) == clip['video_fname'])
    for key in exp_info.keys():
        if isinstance(exp_info[key][ind[0][0]], float) and key != 'participant':
            clip['info'][key] = int(exp_info[key][ind[0][0]])
        elif isinstance(exp_info[key][ind[0][0]], float) and key == 'participant':
            clip['info'][key] = constants.ALL_SUBJECTS[ int(exp_info[key][ind[0][0]]) - 1]
                
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

    
def load_landmarks( temp_landmarks, frame_no):
    """
    Each clip has 300 frames and 68 landmarks per frame.
    temp_landmarks keeps the landmarks for the entire clip. Here I extract 
    the part that relates only this frame (68 points)
    """

    ind_0 = frame_no * constants.NLANDMARKS_PER_FRAME
    ind_f = ind_0 +  constants.NLANDMARKS_PER_FRAME
    landmarks = temp_landmarks[ind_0: ind_f, :]
    
    return landmarks

def get_eyesize( landmarks):
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

        
def get_eyesize_normalized(frame):
    """
    This function computes the mean size of the eyes (mean of right and left)
    for all frames.
    This means that the eyes may be closed or blinking etc.
    """
    eye_size_normalized = {'right': 0,\
                'left' : 0, \
                'mean' : 0}
    
    size_r = frame['eye_size']['right'] / frame['ocular_breadth']['interocular']**2
    size_l = frame['eye_size']['left'] / frame['ocular_breadth']['interocular']**2
    size_m = (size_r + size_l)*0.5

    eye_size_normalized['right'] = size_r
    eye_size_normalized['left'] = size_l
    eye_size_normalized['mean'] = size_m
    
    return eye_size_normalized

def get_eye_aspect_ratio( landmarks):
    """
    This function computes eye aspect ratio. the method is explained in
    
        Real-Time Eye Blink Detection using Facial Landmarks
        Tereza Soukupova and Jan Cech
        CVWW 2016
        
    TODO
    Use the method of the aboive paper to detect blinks
    """
    eye_aspect_ratio = {'right': 0, 'left': 0, 'mean': 0}
    
    eye_aspect_ratio['right'] = (np.linalg.norm(landmarks[37] - landmarks[41]) + \
                    np.linalg.norm(landmarks[38] - landmarks[40]))\
                    / 2 / np.linalg.norm(landmarks[36] - landmarks[39])   
 
    eye_aspect_ratio['left'] = (np.linalg.norm(landmarks[43] - landmarks[47]) + \
                np.linalg.norm(landmarks[44] - landmarks[46]))\
                / 2 / np.linalg.norm(landmarks[42] - landmarks[45])  
                
    eye_aspect_ratio['mean'] =  0.5 * (eye_aspect_ratio['right'] + eye_aspect_ratio['left'])
   
        
    return eye_aspect_ratio

def get_ocular_breadth( landmarks):
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

def load_poses_ACC( t0, frame_no, temp_accmeter): # add
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

    
def get_poses_LM( landmarks):
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
