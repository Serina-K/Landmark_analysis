#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:41:16 2018

@author: zeynep
"""

import file_tools as file_tools
import pickle
import matplotlib.pyplot as plt
import numpy as np

from importlib import reload
import constants
reload(constants)
import preferences
reload(preferences)

"""
This function loads the parameters/variables related to eye features and plots
them with espect to the subject name and experiment name.

The colors represents subjects as
r: ayse
b: buse
m: esra
c: gokhan
k: merve

The markers represent expriment types as
.: slide show
^: wcs game
*: video book

Results:
    
    * Slide show has always the lowest r_open, i.e. eyes least open and the 
    difference of r_open for slide show and the other experiments seem significant
    
    * Slide show has the lowest aopennorm for the videos that I could compute. 
    But there are many videos that I cannot compute. slide show does not look 
    reliable for prediction due to failure of computation of variables
    
    * wcs and video book are about the same. some subject jave higher r_open for 
    wcs and some for video book. Also, the difference is not very big
    
    * for dur_blinks, it seems video book has longer blinks than wcs. slide show
    has most of the time 300 frame long blinks, which means the r_open ratio
    is almost always below the threshold
    
    * If we consider wcs and video book, it is easy to prove the variables some 
    of a kind variationalong the expriment. for slide show, I cannot compute 
    them and it is bad.
    
    *  we can also claim tat video book and wcs do not rely on the recording 
    conditions such as the angle of recording, daily variations, monitor tilt etc
    
    * conclusion is to eliminate slide show from the evalutions
    

"""

clipvar_fnames = file_tools.get_data_fnames(constants.CLIP_VAR_DIR, 'clipvar_')  


counter = 0
colors = ['r', 'b', 'm', 'c', 'k'] # for 5 particpants
markers = ['.', '^', '*'] # 3 experiments

varlist = ['r_open', 'aopennorm', 'dblink_m']

vars_sorted = {}
for var in varlist:
    vars_sorted[var] = {}
    for subject  in preferences.SUBJECTS:
        vars_sorted[var][subject] = {}
        for experiment in range(0,3):
            vars_sorted[var][subject][str(experiment)] = []
        
aopennormsorted = {}
for subject  in preferences.SUBJECTS:
    aopennormsorted[subject] = {}
    for experiment in range(0,3):
        aopennormsorted[subject][str(experiment)] = []


for clipvar_fname in clipvar_fnames:
    with open(str(clipvar_fname), 'rb') as f:
        clip = pickle.load(f)
        if clip['info']['valid']:      
            for i in range(0, len(preferences.SUBJECTS)):

                
                marker = markers[ (clip['info']['experiment'] ) ]
                if clip['info']['participant'] == preferences.SUBJECTS[i]:
                    color = colors[i]
                    
                    for var in varlist:
                        vars_sorted[var][preferences.SUBJECTS[i]][str(clip['info']['experiment'] )].append( \
                               clip[var])

                    
            counter += 1
print('A total of {} clips are valid'.format(counter))

################################################################################
# plot ropen in a sorted manner  
for var in varlist:
    plt.figure()        
    for i, subject  in enumerate(preferences.SUBJECTS):
        for experiment in preferences.EXPERIMENTS :
            vars_sorted[var][subject][str(experiment)] = np.sort((vars_sorted[var][subject][str(experiment)]))
            
            color = colors[i]
            marker = markers[experiment]
            
            plt.plot( vars_sorted[var][subject][str(experiment)] , color+marker+'-')
        
    plt.grid(linestyle='dotted')
    plt.title(var)
################################################################################

    
