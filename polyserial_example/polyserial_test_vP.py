#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:03:13 2018

@author: zeynep
"""
import numpy as np
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
rpy2.robjects.numpy2ri.activate()

polycor = rpackages.importr('polycor')

temp = np.loadtxt('sample_data_v2.txt', delimiter='\t')
 
age = np.asarray( [row[0] for row in temp] )
weight = np.asarray( [row[1] for row in temp] )
runtime = np.asarray( [row[2] for row in temp] )
oxygen = np.asarray( [row[3] for row in temp] )

age_vs_oxy = polycor.polyserial(age, oxygen, ML = True, control = list(), std_err = False, maxcor=.9999, bins=4)
wei_vs_oxy = polycor.polyserial(weight, oxygen, ML = True, control = list(), std_err = False, maxcor=.9999, bins=4)
run_vs_oxy = polycor.polyserial(runtime, oxygen, ML = True, control = list(), std_err = False, maxcor=.9999, bins=4)

print('Age-oxygen\t{}'.format(age_vs_oxy[0]))
print('Weight-oxygen\t{}'.format(wei_vs_oxy[0]))
print('Rutime-oxygen\t{}'.format(run_vs_oxy[0]))