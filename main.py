#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:50:29 2018

@author: zeynep
"""

from clip import *
import numpy as np
import file_tools as file
from os import listdir

import time
import matplotlib.pyplot as plt
import pylab as pl


if __name__ == "__main__":

    start_time = time.time()
    
    exp_info_fname = '../datafiles/experiment_info_codes_v3.txt'
    exp_info = file.readExpInfo(exp_info_fname)
    
    landmark_dir = '../datafiles/landmark/' # change this to your directory
    landmark_fnames = sorted([landmark_dir + f for f in listdir(landmark_dir) if '.dat.npy' in f])
    
    
    accmeter_dir = '../datafiles/accelerometer/' # add
    accmeter_fnames = sorted([accmeter_dir + f for f in listdir(accmeter_dir) if 'acc.npy' in f]) # add
    
    
  
    
    
    
    

    
    
    
    tmp = [] ##add
    #for f, landmark_fname in enumerate(landmark_fnames):
    for i in range(2):
        
        landmark_fname = landmark_fnames[i]  
        accmeter_fname  = accmeter_fnames[i]
        

        
        
        clip = Clip(landmark_fname)
        clip.build_clip(exp_info, landmark_fname)
        tmp.extend(clip.analyze())            
#        clip.get_poses_ACC(accmeter_fname) # add   
        clip.illlustrate()
        
           
#    fig6 = plt.figure()
#    ax2 = fig6.add_subplot(111)
#    mngr = plt.get_current_fig_manager()
#    mngr.window.setGeometry(0,0,640, 480)
   
#    pl.figure(fig6.number)
#   plt.cla()      
#    plt.hist(tmp, bins=50)
#    plt.title("Histgram")
#    plt.xlabel("x")
#    plt.ylabel("y")
#    plt.show()
    

    elapsed_time = time.time() - start_time
    #print('Time elapsed  %2.2f sec' %elapsed_time)
    
    
#    print(tmp)