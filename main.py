#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:50:29 2018

@author: zeynep
"""

import warnings
warnings.simplefilter("once", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
import time

#from clip import *

from importlib import reload

import constants
reload(constants)
import clip_builder
import pscor_tools
import blink_tools
import kde_model
import estimator_tools
import prob_est

if __name__ == "__main__":

    start_time = time.time()
    
    #clip_builder.build_all()

    #blink_tools.detect()
    pscor_tools.calculate()
    #kde_model.calculate()

    #estimator_tools.estimate()
    #prob_est.get_eng_prob()
        
    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time)
    
