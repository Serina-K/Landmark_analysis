#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 22:50:50 2018

@author: zeynep
"""
import sys
sys.path.insert(0, '../file_tools')
sys.path.insert(0, '../constants')

import file_tools as file_tools

import csv
import numpy as np

import constants

import time
start_time = time.time()

# Takes about 15 min on mac

exp_info = file_tools.readExpInfo('../'+constants.EXP_INFO_FNAME)

for i, video_fname in enumerate(exp_info['video_fname']):
    
    participant_code = exp_info['participant'][i]
    
    if int(participant_code) is 1:
        participant = 'ayse'
    elif int(participant_code) is 2:
        participant = 'buse'
    elif int(participant_code) is 3:
        participant = 'esra'
    elif int(participant_code) is 4:
        participant = 'gokhan'
    elif int(participant_code) is 5:
        participant = 'merve'
        
    exp_no = exp_info['experiment'][i]
    
    acc_data_fname = '../../datafiles/raw_accelerometer/'+\
    participant + '_exp' + str(int(exp_no)) + '_acc.csv'
    
    print(acc_data_fname)
    
    temp = np.loadtxt(open(acc_data_fname, "rb"), delimiter=",", skiprows=1)
    
    t0_unix = exp_info['clip_t0_unix'][i]
    tf_unix = exp_info['clip_t0_unix'][i] + 10
    
    ind = np.logical_and(temp[:,0] >= t0_unix, temp[:,0]< tf_unix)
    
    out_fname = '../../datafiles/accelerometer/' + video_fname + '_acc.npy'
    np.save(out_fname, temp[ind, :])

elapsed_time = time.time() - start_time    
print('Time elapsed  %2.2f sec' %elapsed_time)

    
