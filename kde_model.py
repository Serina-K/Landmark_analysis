#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:01:05 2018

@author: zeynep
"""

import numpy as np
import pickle
import file_tools as file_tools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
# on mac import from sklearn.grid_search, on ubuntu from sklearn.model_selection 
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV


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
                    
def calculate():
    
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
    
    
    opt_kdes = {}
    for var in varlist:  
        opt_kdes[var] = {}
        plt.figure()
        
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
                
        for label in ulabels:
            
            opt_kdes[var][label] = []

            temp_var_values = np.asarray(var_values[label])[:, np.newaxis]
            xmax = np.max(var_values[label])
            # for plotting on extended range [0,1]
            X_plot_ext = np.linspace(0, 1, 1000)[:, np.newaxis]


            if len(temp_var_values):
                #----------------------------------------------------------------------                
                # with sklearn  
                bandwidths = np.linspace(0.001, xmax, 1000)
                grid = GridSearchCV(KernelDensity(kernel='gaussian'),\
                        {'bandwidth': bandwidths})
                
                grid.fit(np.array(temp_var_values))
                best_bw = grid.best_params_['bandwidth']
                
                print('{}\t{}\t{}'.format(var, label, best_bw))
                
                kde = KernelDensity(kernel='gaussian', bandwidth=best_bw).fit(temp_var_values)
                opt_kdes[var][label]  = kde
                
                log_dens = kde.score_samples(X_plot_ext)
                norm_dens = (np.exp(log_dens)) / np.sum(np.exp(log_dens))
                plt.plot(X_plot_ext[:, 0], norm_dens, '-',
                        label="kernel = '{0}'".format('gaussian'))
        plt.legend(['1', '2', '3', '4', '5'])
        plt.xlabel(var)
        plt.grid(b=True, linestyle='--', linewidth=0.5)
        
    # Saving the objects:
    with open('opt_kdes.pkl', 'wb') as f:  
        pickle.dump(opt_kdes, f)

#    # Getting back the objects:
#    with open('opt_kdes.pkl', 'rb') as f: 
#        opt_kdes = pickle.load(f)


