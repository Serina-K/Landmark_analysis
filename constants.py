#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 15:52:14 2018

@author: zeynep
"""

EXP_INFO_FNAME = '../datafiles/experiment_info_codes_v6.txt'
LANDMARK_DIR = '../datafiles/landmark/' 
ACCMETER_DIR = '../datafiles/accelerometer/'
CLIP_VAR_DIR = '../datafiles/clip_vars/' 

ALL_SUBJECTS = ['ayse','buse', 'esra','gokhan','merve']

NSUBJECTS = 5 # number of participants


# each video has 300 frames and each frame has 68 landmarks
NFRAMES = 300
FPS = 30
NLANDMARKS_PER_FRAME = 68
NLANDMARKS_PER_CLIP = NFRAMES * NLANDMARKS_PER_FRAME

# you may cross-check with dimensions of exp_info, landmark_fnames etc
NCLIPS = 214

NBINS_EYE_ASPECT_RATIO = 50


