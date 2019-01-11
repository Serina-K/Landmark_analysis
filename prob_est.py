#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 10:43:41 2018

@author: zeynep
"""


import numpy as np
import pickle
import file_tools as file_tools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from sklearn.neighbors import KernelDensity
# on mac import from sklearn.grid_search, on ubuntu from sklearn.model_selection 
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt


from importlib import reload

import constants
reload(constants)
import preferences
reload(preferences)

    
def init_mats(clipvar_fnames):
    
    labels = {}
    values = {}
    
    # dblink_m', 'aclosednorm', 'r_closed', 'aopennorm', 'a_both', 
    #'intocu_m', 'r_both', 'r_open', 'num_blinks', 'biocul_m', 'nblinks'
    varlist = preferences.FEATS
    
    for participant in preferences.SUBJECTS:
        labels[participant] = {}
        values[participant] = {}
        for annotator in preferences.ANNOTATORS:
            labels[participant][annotator] = []
    
        for var in varlist:                    
            values[participant][var] = []
        
    return varlist, labels, values
               
def get_eng_prob():
    
    plt.figure()
    
    clipvar_fnames = file_tools.get_data_fnames(constants.CLIP_VAR_DIR, 'clipvar_')  
    
    varlist, labels, values = init_mats(clipvar_fnames)
                
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
                            
    temp1 = list(labels.values())
    temp2 = [list(sublist.values()) for sublist in temp1]
    flat_list = [item for sublist in temp2 for subsublist in sublist for item in subsublist]
    ulabels = np.unique(flat_list)
    
    prob_eng = {}
    for label in ulabels:   
        prob_eng[label] = []
        
        
    # Getting back the objects:
    with open('opt_kdes.pkl', 'rb') as f: 
        opt_kdes = pickle.load(f)
    
    for var in varlist:  
        
        var_values = {}
        for label in ulabels:
            var_values[label] = []
                
        for participant in preferences.SUBJECTS:
            
            temp_values_v0 = values[participant][var]
            temp_labels_v0 = labels[participant][preferences.GT_ANNOTATOR] 
    
            temp_values = np.asarray([x for x,y in zip(temp_values_v0, temp_labels_v0) if np.isfinite(x)])
            temp_labels =  np.asarray([y for x,y in zip(temp_values_v0, temp_labels_v0) if np.isfinite(x)])
            
            #  temp_labels is between 1 and 5
            for j, label in enumerate(temp_labels):
                var_values[label].append(temp_values[j])
                
        for gt_label in ulabels:

            temp_var_values = np.asarray(var_values[gt_label])[:, np.newaxis]
            
            # consider two extremes fully engaged (1) and completely disengaged (5)
            
            # fully engaged, 1
            est_label_eng = np.min(ulabels)
            log_dens_eng = opt_kdes[var][est_label_eng].score_samples(temp_var_values)
            norm_dens_eng = (np.exp(log_dens_eng)) / np.sum(np.exp(log_dens_eng))
            
            # completely disengaged, 5
            est_label_diseng = np.max(ulabels)
            log_dens_diseng = opt_kdes[var][est_label_diseng].score_samples(temp_var_values)
            norm_dens_diseng = (np.exp(log_dens_diseng)) / np.sum(np.exp(log_dens_diseng))

            # normalize total prob to 1
            p_tot = np.add(norm_dens_eng, norm_dens_diseng)
            norm_dens_eng = np.divide(norm_dens_eng, p_tot)
            norm_dens_diseng = np.divide(norm_dens_diseng, p_tot)
            
            prob_eng[gt_label].extend(norm_dens_eng)
        
        xs, ys = [], []
        print('{}'.format(var))
        print('GT\t p(eng)')
        for gt_label in ulabels:   
            xs.append(gt_label)
            ys.append(np.mean(prob_eng[gt_label]))
            print('{}\t{}'.format(gt_label, np.mean(prob_eng[gt_label])))
                         


        plt.plot(xs, ys, '.-', label=var)
        plt.grid(linestyle='dotted')