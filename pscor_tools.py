#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 09:30:11 2018

@author: zeynep
"""
import numpy as np
import pickle
import file_tools as file_tools

import matplotlib.pyplot as plt

import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
rpy2.robjects.numpy2ri.activate()

polycor = rpackages.importr('polycor', lib_loc='/usr/local/lib/R/3.5/site-library')

from importlib import reload

import constants
reload(constants)
import preferences
reload(preferences)


def init_mats(clipvar_fnames):
    
    labels = {}
    pcscor = {}
    values = {}
    
    # dblink_m', 'aclosednorm', 'r_closed', 'aopennorm', 'a_both', 
    #'intocu_m', 'r_both', 'r_open', 'num_blinks', 'biocul_m', 'nblinks'
    varlist = preferences.FEATS
    
    
    for participant in preferences.SUBJECTS:
        labels[participant] = {}
        pcscor[participant] = {}
        values[participant] = {}
        for annotator in preferences.ANNOTATORS:
            labels[participant][annotator] = []
    
        for var in varlist:                    
            pcscor[participant][var] = {}
            values[participant][var] = []
            for annotator in preferences.ANNOTATORS:
                pcscor[participant][var][annotator] = []
    
    return varlist, pcscor, labels, values
                    
def calculate():
    
    clipvar_fnames = file_tools.get_data_fnames(constants.CLIP_VAR_DIR, 'clipvar_')  
    
    varlist, pcscor, labels, values = init_mats(clipvar_fnames)
                
    for clipvar_fname in clipvar_fnames:
        with open(str(clipvar_fname), 'rb') as f:
            clip = pickle.load(f)
            if ((clip['info']['participant'] in preferences.SUBJECTS) and\
                clip['info']['experiment'] in preferences.EXPERIMENTS and\
                clip['info']['valid'] ):
                
                for participant in preferences.SUBJECTS:
                    if ( clip['info']['participant'] ==  participant):
                        
                        for var in varlist:   
                            values[participant][var].append(clip[var])  
                            
                        for annotator in preferences.ANNOTATORS:
                            labels[participant][annotator].append(clip['info'][annotator])                                  

 
    for var in varlist:  
        plt.figure()
        for participant in preferences.SUBJECTS:
            temp_values_v0 = values[participant][var]                 
            for annotator in preferences.ANNOTATORS:
                temp_labels_v0 = labels[participant][annotator] 
        
                temp_values = np.asarray([x for x,y in zip(temp_values_v0, temp_labels_v0) if np.isfinite(x)])
                temp_labels =  np.asarray([y for x,y in zip(temp_values_v0, temp_labels_v0) if np.isfinite(x)])
                                    
                if len(temp_values):
                    pcscor[participant][var][annotator] = \
                    polycor.polyserial(temp_values, temp_labels, ML = True, \
                                       control = list(), std_err = False, maxcor=.9999, bins=4) 
                if annotator == 'annotator1':
                    plt.plot(temp_values, temp_labels, 'b.')
        plt.xlabel(var)

    for var in varlist:
        print('{}\t'.format(var), end='', flush=True)
        for participant in preferences.SUBJECTS:
            for annotator in preferences.ANNOTATORS:
                if len(pcscor[participant][var][annotator]):
                    print('{:.2f} '.format( np.asarray(pcscor[participant][var][annotator][0])), end='', flush=True)
        print(' ')
        
    for var in varlist:
        print('{}\t'.format(var), end='', flush=True)
        for participant in preferences.SUBJECTS:
            for annotator in preferences.ANNOTATORS:
                if len(pcscor[participant][var][annotator]):
                    if np.sign(np.asarray(pcscor[participant][var][annotator][0])) > 0:
                        print('+ ', end='', flush=True)
                    elif np.sign(np.asarray(pcscor[participant][var][annotator][0])) < 0:
                        print('- ', end='', flush=True)
                    else:
                        print('0 ', end='', flush=True)
                            
                            
        print(' ')            
       
