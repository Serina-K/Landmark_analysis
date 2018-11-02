#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 16:33:01 2018

@author: zeynep
"""

import file_tools as file_tools
import time

import pickle
from pathlib import Path

import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import mixture


import constants
from importlib import reload
reload(constants)

class Blink_detector():
    """
    Blink detector pools aspect rations of individual and applies GMM on each 
    set. 
    """
    def __init__(self, illustrate):
        
        self.illustrate = illustrate
        self.clipvar_fnames = file_tools.get_data_fnames(constants.CLIP_VAR_DIR, 'clipvar_')  

        self.aspect_ratios_v0 = {}
        self.blink_thresholds = {}
        for participant_no in range(0,constants.NSUBJECTS):
            self.aspect_ratios_v0[str(participant_no)] = []
            self.blink_thresholds[str(participant_no)] = 0
            
    def detect(self):
        self.estimate_thresholds()
        self.set_thresholds_etc()
        
        
        
    def estimate_thresholds(self):
        """
        Pools the eye aspect ratios from each subject to a different array. Then
        fit a GMM with  2 componenets and returns the intersection point (its x-
        coordinate) as the blink threshold.
        
        """
        print('Estimating blink thresholds...')        

        for clipvar_fname in self.clipvar_fnames:
            with open(str(clipvar_fname), 'rb') as f:
                clip = pickle.load(f)
                if clip['info']['valid']:            
                    for frame in clip['frames']:
                        self.aspect_ratios_v0[\
                                         str(int(clip['info']['participant']))].\
                                         append(frame['eye_aspect_ratio']['mean'])
           

        for participant_no in range(0, constants.NSUBJECTS):
            self.blink_thresholds[str(participant_no)] = \
            self.gmm_on_eye_aspect_ratios(participant_no, False)
          
    def set_thresholds_etc(self):
        
        for clipvar_fname in self.clipvar_fnames:
                with open(str(clipvar_fname), 'rb') as f:
                    clip = pickle.load(f)
                    if clip['info']['valid']:  
                        
                        eye_state = [] # append 1 for open, 0 for closed
                        nframes_eye_open, nframes_eye_closed = 0, 0
                        aopen_sum, aclosed_sum = 0, 0
                        nblink = 0
                        
                        
                        R0 = self.blink_thresholds[str(int(clip['info']['participant']))] 
                        clip['blink_threshold'] = R0
                        
                        for frame in clip['frames']:
                            if frame['eye_aspect_ratio']['mean'] <= clip['blink_threshold']:
                                eye_state.append(0)
                                nframes_eye_closed += 1
                                aclosed_sum += frame['eye_size']['mean']
                            else:
                                eye_state.append(1)
                                nframes_eye_open += 1
                                aopen_sum += frame['eye_size']['mean']
                        
                        nblink = np.sum(np.diff(eye_state) == -1)
                        dur_closed_tot = np.sum(np.asarray(eye_state) == 0)
                        
                        if not dur_closed_tot:
                            dblink = 0
                        else:
                            if not nblink:
                                dblink = 300
                            else:
                                dblink = dur_closed_tot / nblink
                                
                        if np.isnan(dblink) or np.isinf(dblink):
                            print('problem {} {}'.format(np.isnan(dblink), np.isinf(dblink)))
                            
                        
                        if nframes_eye_open > 0:
                            aopen = aopen_sum / nframes_eye_open
                        else:
                            aopen = np.nan
                            
                        if nframes_eye_closed > 0:
                            aclosed = aclosed_sum / nframes_eye_closed
                        else:
                            aclosed = np.nan
                                
                        clip['nblinks'] = nblink
                        clip['aopen'] = aopen
                        clip['aclosed'] = aclosed
                        clip['dblink'] = dblink

                        
                        
                        with open(str(clipvar_fname), 'wb') as f:
                            pickle.dump(clip, f, pickle.HIGHEST_PROTOCOL)
                        
                        
                        
        
    def gmm_on_eye_aspect_ratios(self, participant_no, plot_mixtures=True):
        """
        The intersection of the 2 components of the Gaussian Mixture Model is 
        considered to the threshold R0.
        """
        aspect_ratios = self.aspect_ratios_v0[str(participant_no)]     
        
        if (plot_mixtures):
            plt.figure() 
            n, bins, patches = plt.hist(aspect_ratios, constants.NBINS_EYE_ASPECT_RATIO, density=1)
            
        g = mixture.GaussianMixture(n_components=2, covariance_type='spherical')
        aspect_ratios = np.asarray(aspect_ratios)
        aspect_ratios = aspect_ratios.reshape((-1, 1))
        
        g.fit(aspect_ratios)
    
        weights = g.weights_
        means = g.means_
        covars = g.covariances_

        D = aspect_ratios.ravel()
        xmin = 0#D.min()
        xmax = 0.5#D.max()
        x = np.linspace(xmin,xmax,1000)
        
        mean0 = means[0]
        sigma0 = math.sqrt(covars[0])
        gauss0 = weights[0]*stats.norm.pdf(x, mean0, sigma0)
       
        mean1 = means[1]
        sigma1 = math.sqrt(covars[1])
        gauss1 = weights[1]*stats.norm.pdf(x, mean1, sigma1)
       
        gaussT = gauss0 + gauss1 
        # we assume the intersection of the two curves appears between 
        # 0.3 to 0.6 of the x-axis
        R0 = np.argmin(abs(gauss0[int(0.3*len(x)) : int(0.6*len(x))] -\
                             gauss1[int(0.3*len(x)) : int(0.6*len(x))])) +\
                             int(0.3*len(x))
                             
        #######################################################################
        print('-------------------------------------------')

        print('Participant no: {}'.format(int(participant_no)))
        print('Parameters of Gaussian mixture components: ')
        print('Weights:\t{:.4f}\t{:.4f}'.format(weights[0], weights[1]))
        print('Means:  \t{:.4f}\t{:.4f}'.format(means[0][0], means[1][0]))
        print('Stds:   \t{:.4f}\t{:.4f}'.format(covars[0], covars[1]))
        print('Threshold:\t{:.4f}'.format(x[R0]))
        
        if (self.illustrate):
            plt.plot(x, gaussT, c='black')
            plt.plot(x, gauss0, c='red')
            plt.plot(x, gauss1, c='blue')
            
            plt.plot(x[R0], gauss0[R0], 'o')
            print('Threshold location: {:.4f}'.format(x[R0]))
            
            plt.title('Participant no: ' + str(participant_no))
            plt.xlabel('R (eye aspect ratio)')
            plt.ylabel('Histogram')
            plt.grid(b=True, linestyle='--', linewidth=0.5)
            axes = plt.gca()
            axes.set_xlim([0, 0.5])
            axes.set_ylim([0, 15])
            plt.show()
        
        return x[R0]
    
