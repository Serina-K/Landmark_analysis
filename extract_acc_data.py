#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 12:37:23 2018

This function is meant to run only once. It extracts the accelerometer data from
the time intervals that correspond to the video clips (of 10 sec) and save it 
as a data file.

@author: zeynep
"""

from clip import *
import numpy as np
import file_tools as file
from os import listdir

import time

if __name__ == "__main__":

    start_time = time.time()
    
    exp_info_fname = '../datafiles/experiment_info_codes_v3.txt'
    exp_info = file.readExpInfo(exp_info_fname)
    
    for e, info in enumerate(exp_info['video_fname']):
        
        participant_code = exp_info['participant'][e]
        
        if participant_code == 1:
            participant_name = 'ayse'
        elif  participant_code == 2:
            participant_name = 'buse'
        elif  participant_code == 3:
            participant_name = 'esra'
        elif  participant_code == 4:
            participant_name = 'gokhan'
        elif  participant_code == 5:
            participant_name = 'merve'
        else:
           print('Problem with participant code')     
    
        exp_no = exp_info['experiment'][e]
        
        clip_t0 = int(exp_info['clip_t0_unix'][e])
        clip_tf = clip_t0 + 10 # the duration of all clips is 10 sec 
        
        in_fname = '../datafiles/accelerometer/' + participant_name + '_exp' +  str(int(exp_no)) + '_acc.csv'
           
        data_tot = np.genfromtxt(in_fname, delimiter=', ', skip_header=True)
                             
        ind = np.logical_and((clip_t0 < data_tot[:,0]),  (data_tot[:,0] < clip_tf))
        acc_data = data_tot[ind, 1:4]
        
        out_fname = '../datafiles/accelerometer/' + info + '_acc'

        np.save(out_fname, acc_data)
        
    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time)