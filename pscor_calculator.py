#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 09:30:11 2018

@author: zeynep
"""
import numpy as np
import pickle
import file_tools as file_tools

import constants
from importlib import reload
reload(constants)

import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
rpy2.robjects.numpy2ri.activate()

polycor = rpackages.importr('polycor', lib_loc='/usr/local/lib/R/3.5/site-library')

import constants
from importlib import reload
reload(constants)

class PScor_calculator():
    """
    TODO 
    """
    def __init__(self):
        self.pcor_anno1, self.pcor_anno2 = [], []
        self.annotator1, self.annotator2, self.vals = [], [], []
        
        self.clipvar_fnames = file_tools.get_data_fnames(constants.CLIP_VAR_DIR, 'clipvar_')  
        
        with open(str(self.clipvar_fnames[0]), 'rb') as f:
            dummyclip = pickle.load(f)
        self.keys = list(dummyclip.keys())
        
                    
    def calculate(self):
        for t in range(0, constants.NSUBJECTS):    
            partipant_no = t
            
            print('--------------------------------')
            print('Participant: {}'.format(partipant_no))
            print('Var.\tAnnot-1\tAnnot-2'.format(partipant_no))
        
            for key in self.keys:
                self.pcor_anno1, self.pcor_anno2 = [], []
                self.annotator1, self.annotator2, self.vals = [], [], []
                
                for clipvar_fname in self.clipvar_fnames:
                    with open(str(clipvar_fname), 'rb') as f:
                        clip = pickle.load(f)
                        if ((int(clip['info']['participant']) is t ) and\
                            clip['info']['valid'] ):
                            if (isinstance(clip[key], np.int64) or isinstance(clip[key], float) and\
                                (key != 'blink_threshold')):
                                self.annotator1.append(clip['info']['annotator1'])
                                self.annotator2.append(clip['info']['annotator2'])
                                self.vals.append(clip[key])  
                    
            
                temp_vals = np.asarray([x for x,y in zip(self.vals, self.annotator1) if np.isfinite(x)])
                temp_anno1 = np.asarray([y for x,y in zip(self.vals, self.annotator1) if np.isfinite(x)])
                temp_anno2 = np.asarray([y for x,y in zip(self.vals,self. annotator2) if np.isfinite(x)])
            
                if len(temp_vals):
                    self.pcor_anno1 = polycor.polyserial(temp_vals, temp_anno1, ML = True, control = list(), std_err = False, maxcor=.9999, bins=4) 
                    self.pcor_anno2 = polycor.polyserial(temp_vals, temp_anno2, ML = True, control = list(), std_err = False, maxcor=.9999, bins=4) 
            
                #run_vs_oxy = polycor.polyserial(runtime, oxygen, ML = True, control = list(), std_err = False, maxcor=.9999, bins=4)
        
                if len(self.pcor_anno1) and len(self.pcor_anno2):
                    print('{}\t{:.4f}\t{:.4f}'.format(key, np.asarray(self.pcor_anno1[0]), np.asarray(self.pcor_anno2[0])))
#
#        
#                
