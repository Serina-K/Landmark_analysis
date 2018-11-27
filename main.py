#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:50:29 2018

@author: zeynep
"""

import warnings
warnings.simplefilter("once", UserWarning)

from clip_builder import *
from blink_detector import *
from pscor_calculator import *
from label_estimator import *

import time


# if the constants/preferences are not refreshed, take a long way and reload
import constants
from importlib import reload
reload(constants)


import clip_tools
from importlib import reload
reload(clip_tools)



if __name__ == "__main__":

    start_time = time.time()
    
    #builder = Clip_builder()
    #builder.build_all()
    
#    detector = Blink_detector(illustrate = False)
#    detector.detect()
#    
#    calculator = PScor_calculator()
#    calculator.calculate()
    
    estimator = Label_estimator()
    estimator.estimate()

    
    

        
    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time)
    
