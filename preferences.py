#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:57:30 2018

@author: zeynep
"""

# options 'ayse','buse', 'merve', 'esra','gokhan'
SUBJECTS = [ 'esra','gokhan', 'merve']

# 0: slide show
# 1: wcs game
# 2: video book
EXPERIMENTS = [ 1, 2]

ANNOTATORS = ['annotator1', 'annotator2']
GT_ANNOTATOR = 'annotator2'


NORMALIZATION_FACTOR_OCULAR_DIST = 300 # frame width is 1280
NORMALIZATION_FACTOR_DBLINK = 20
NORMALIZATION_FACTOR_NBLINK = 10



# dblink_m', 'aclosednorm', 'r_closed', 'aopennorm', 'a_both', 
#'intocu_m', 'r_both', 'r_open', 'num_blinks', 'biocul_m', 'roll_m', 'yaw_m'
FEATS = [ 'dblink_m','r_open','aopennorm', 'intocu_m', 'biocul_m','roll_m', 'num_blinks']

CLASSIFIER = 'knn'

N_NEIGHBORS = 1

N_MIXTURE_COMPONENTS = 2
