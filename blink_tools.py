#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 16:33:01 2018

@author: zeynep
"""

import file_tools as file_tools

import pickle
from pathlib import Path
from scipy.interpolate import interp1d

import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import mixture
from scipy import optimize

from importlib import reload

import constants
reload(constants)

import preferences
reload(preferences)

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """


    if window_len<3:
        return x


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

"""
Blink detector pools aspect rations of individual and applies GMM on each 
set. 
"""
def init_mats(illustrate):
    
    illustrate = illustrate
    clipvar_fnames = file_tools.get_data_fnames(constants.CLIP_VAR_DIR, 'clipvar_')  

    aspect_ratios_v0 = {}
    aspect_smooth = {}
    blink_thresholds = {}
    for participant in preferences.SUBJECTS:
        aspect_ratios_v0[participant] = []
        aspect_smooth[participant] = []
        blink_thresholds[participant] = 0
        
    return clipvar_fnames, aspect_ratios_v0, aspect_smooth, blink_thresholds
    
    
    
def estimate_thresholds(clipvar_fnames, blink_thresholds, aspect_ratios_v0, aspect_smooth):
    """
    Pools the eye aspect ratios from each subject to a different array. Then
    fit a GMM with  2 componenets and returns the intersection point (its x-
    coordinate) as the blink threshold.
    
    """
    print('Estimating blink thresholds...')        

    for clipvar_fname in clipvar_fnames:
        with open(str(clipvar_fname), 'rb') as f:
            clip = pickle.load(f)
            if (clip['info']['valid'] and \
                clip['info']['participant'] in preferences.SUBJECTS and\
                clip['info']['experiment'] in preferences.EXPERIMENTS):  
                
                for frame in clip['frames']:
                    aspect_ratios_v0[\
                                     clip['info']['participant']].\
                                     append(frame['eye_aspect_ratio']['mean'])
       

    for participant in preferences.SUBJECTS:
        blink_thresholds[participant],  tempx, tempy = optimize_gmm_on_eye_aspect_ratios(aspect_ratios_v0, participant, False, True)
        aspect_smooth[participant] = tempy.copy()
        
        #optimize_gmm_on_eye_aspect_ratios(aspect_ratios_v0, participant, False, True)
        
    plot_all_aspect_ratios(tempx, aspect_smooth, True)
        
    return blink_thresholds
      
def set_thresholds_etc(clipvar_fnames, blink_thresholds):
    
    for clipvar_fname in clipvar_fnames:
            with open(str(clipvar_fname), 'rb') as f:
                clip = pickle.load(f)
                
                if clip['info']['valid'] and\
                (clip['info']['participant'] in preferences.SUBJECTS) and\
                (clip['info']['experiment'] in preferences.EXPERIMENTS):  
                    
                    eye_state = [] # append 1 for open, 0 for closed
                    nframes_eye_open, nframes_eye_closed = 0, 0
                    aopennorm_sum, aclosednorm_sum = 0, 0
                    intocu_s , biocul_s = 0,0
                    r_open_sum, r_closed_sum = 0, 0
                    nblink = 0
                    RR_s = 0 # this is aspect ratio without paying regard to being open or closed
                    AA_s = 0 # this is eye size without paying regard to being open or closed
                    roll_s = 0 # assuming the head rest at the same tilt all the time
                    yaw_s = 0
                    
                    
                    R0 = blink_thresholds[clip['info']['participant']] 
                    clip['blink_threshold'] = R0
                    
                    for frame in clip['frames']:
                        intocu_s = intocu_s + frame['ocular_breadth']['interocular']
                        biocul_s = biocul_s + frame['ocular_breadth']['biocular']
                        RR_s += frame['eye_aspect_ratio']['mean']
                        AA_s += frame['eye_size_normalized']['mean']
                        roll_s +=  frame['pose_ACC']['roll']
                        yaw_s += frame['pose_ACC']['yaw']
                        
                        if frame['eye_aspect_ratio']['mean'] <= clip['blink_threshold']:
                            eye_state.append(0) # eye closed
                            nframes_eye_closed += 1
                            aclosednorm_sum += frame['eye_size_normalized']['mean']
                            r_closed_sum += frame['eye_aspect_ratio']['mean']
                        else:
                            eye_state.append(1)
                            nframes_eye_open += 1 # eye open
                            aopennorm_sum += frame['eye_size_normalized']['mean']
                            r_open_sum += frame['eye_aspect_ratio']['mean']
                    
                    nblink = np.sum(np.diff(eye_state) == -1)
                    nblink_norm = nblink / preferences.NORMALIZATION_FACTOR_DBLINK
                    dur_closed_tot = np.sum(np.asarray(eye_state) == 0)
                    
                    if not dur_closed_tot:
                        dblink_m = 0
                    else:
                        if not nblink:
                            dblink_m = constants.NFRAMES / preferences.NORMALIZATION_FACTOR_DBLINK
                        else:
                            dblink_m = dur_closed_tot / nblink / preferences.NORMALIZATION_FACTOR_DBLINK
                            
                    if np.isnan(dblink_m) or np.isinf(dblink_m):
                        print('problem {} {}'.format(np.isnan(dblink_m), np.isinf(dblink_m)))
                        
                    
                    if nframes_eye_open > 0:
                        aopennorm = aopennorm_sum / nframes_eye_open
                        r_open = r_open_sum / nframes_eye_open
                    else:
                        aopennorm = np.nan
                        r_open = np.nan

                        
                    if nframes_eye_closed > 0:
                        aclosednorm = aclosednorm_sum / nframes_eye_closed
                        r_closed = r_closed_sum / nframes_eye_closed

                    else:
                        aclosednorm = np.nan
                        r_closed = np.nan
                        
                    intocu_m = intocu_s / (nframes_eye_closed + nframes_eye_open) / \
                    preferences.NORMALIZATION_FACTOR_OCULAR_DIST
                    
                    biocul_m = biocul_s / (nframes_eye_closed + nframes_eye_open)/ \
                    preferences.NORMALIZATION_FACTOR_OCULAR_DIST
                    
                    RR_m = RR_s / (nframes_eye_closed + nframes_eye_open)
                    AA_m = AA_s / (nframes_eye_closed + nframes_eye_open)
                    roll_m = roll_s / (nframes_eye_closed + nframes_eye_open)
                    yaw_m = yaw_s / (nframes_eye_closed + nframes_eye_open)
                    
                    clip['num_blinks'] = nblink_norm # so that it is [0, 1]
                    clip['aopennorm'] = aopennorm
                    clip['aclosednorm'] = aclosednorm
                    clip['r_open'] = r_open
                    clip['r_closed'] = r_closed
                    clip['dblink_m'] = dblink_m
                    clip['intocu_m'] = intocu_m
                    clip['biocul_m'] = biocul_m
                    clip['r_both'] = RR_m
                    clip['a_both'] = AA_m
                    clip['roll_m'] = roll_m
                    clip['yaw_m'] = yaw_m

                    
                    
                    with open(str(clipvar_fname), 'wb') as f:
                        pickle.dump(clip, f, pickle.HIGHEST_PROTOCOL)
                    
                    
def plot_all_aspect_ratios(bin_mids, aspect_smooth, plot_all_together):

    plt.figure() 
    for participant in preferences.SUBJECTS:
        plt.plot(bin_mids, aspect_smooth[participant])     
    plt.title('Histogram for aspect ratio for each participant')
    plt.xlabel('Aspect ratio')
    plt.ylabel('N observations')
    plt.grid(linestyle='--', linewidth=.5)               
    
def gmm_on_eye_aspect_ratios(aspect_ratios_v0, participant,report_results=False, plot_mixtures=True):
    """
    The intersection of the 2 components of the Gaussian Mixture Model is 
    considered to the threshold R0.
    """
    aspect_ratios = aspect_ratios_v0[participant]     
    
    if (plot_mixtures):
        plt.figure() 
        n, bins, patches = plt.hist(aspect_ratios, constants.NBINS_EYE_ASPECT_RATIO, density=1)
        
    g = mixture.GaussianMixture(n_components=preferences.N_MIXTURE_COMPONENTS, \
                                covariance_type='full', \
                                tol = 10**-5, \
                                max_iter=500)
    
    aspect_ratios = np.asarray(aspect_ratios)
    aspect_ratios = aspect_ratios.reshape((-1, 1))
    #######################################################################

    g.fit(aspect_ratios)

    weights = g.weights_
    means = g.means_
    covars = g.covariances_

    D = aspect_ratios.ravel()
    xmin = 0#D.min()
    xmax = 0.5#D.max()
    x = np.linspace(xmin,xmax,1000)
    
    gauss_comps = []
    gaussT = np.zeros(1000)
    
    for i in range(0, preferences.N_MIXTURE_COMPONENTS):
        gauss_comps.append(weights[i]*stats.norm.pdf(x, means[i], math.sqrt(covars[i])))
        gaussT = np.add(gaussT, gauss_comps[i] )
        
        if i > 0:
            R0 = np.argmin(abs(gauss_comps[i-1][int(0.3*len(x)) : int(0.6*len(x))] -\
                               gauss_comps[i][int(0.3*len(x)) : int(0.6*len(x))])) +\
                               int(0.3*len(x))
  
    #######################################################################
    
    bin_mids = (bins[1:] + bins[:-1]) / 2 # midpoint of  bins
    
#    ind_dom = np.argmax(weights) # the index of the dominant componant
#    gauss_dominant = weights[ind_dom]*stats.norm.pdf(bin_mids, means[ind_dom], math.sqrt(covars[ind_dom]))
#    gauss_residual = n - gauss_dominant
#    plt.plot(bin_mids, gauss_residual, 'm')
       
    window_len=11
    t_crop = int((window_len -1)/2)
    y = smooth(n, window_len, window='hanning')
    y_smooth = y[t_crop : -t_crop]
    m1_ind = np.min(np.where(np.sign(np.diff(y_smooth)) == -1))
    m1 = bin_mids[m1_ind] # first local maxima
    m2_ind = np.argmax(y_smooth) # global maxima
    m2 = bin_mids[m2_ind] # first local maxima
    
    plt.plot(bin_mids, y_smooth, 'k--')
    plt.plot(m1, y_smooth[m1_ind], 'o')
    plt.plot(m2, y_smooth[m2_ind], 'o')
         
#    f_cubic   = interp1d(bin_mids, n, kind='cubic')
#    binx = np.linspace(min(bin_mids),max(bin_mids),1000)
#    nx = f_cubic(binx)
#    plt.plot(binx, nx, 'c')
#    
#    gauss_compsx = []
#    gaussTx = np.zeros(1000)
#    for i in range(0, preferences.N_MIXTURE_COMPONENTS):
#        gauss_compsx.append(weights[i]*stats.norm.pdf(binx, means[i], math.sqrt(covars[i])))
#        gaussTx = np.add(gaussTx, gauss_compsx[i] )
        

    #######################################################################
    
    
    if (report_results):
        print('-------------------------------------------')
        print('Participant:\t{}'.format(participant))
        print('Number of mixture components:\t{}'.format(preferences.N_MIXTURE_COMPONENTS))
        print('Parameters of mixture components: ')
        print('Weight\tMean\tStd')
        for i in range(0, preferences.N_MIXTURE_COMPONENTS):
            print("{0:.3f}\t{0:.3f}\t{0:.3f}".format(weights[i], means[i], covars[i]))
            
        print('Threshold:\t{:.4f}'.format(x[R0]))
    #######################################################################

    if (plot_mixtures):
        plt.plot(x, gaussT, c='black')
        for i in range(0, preferences.N_MIXTURE_COMPONENTS):
            plt.plot(x, gauss_comps[i])
       
        
        plt.plot(x[R0], gauss_comps[0][R0], 'o')
                
        plt.title('Participant no: ' + participant)
        plt.xlabel('R (eye aspect ratio)')
        plt.ylabel('Histogram')
        plt.grid(b=True, linestyle='--', linewidth=0.5)
        axes = plt.gca()
        axes.set_xlim([0, 0.5])
        axes.set_ylim([0, 15])
        plt.show()
    
    return x[R0], bin_mids, y_smooth

def eval_norm(x, m, s):
    """
    Evaluate normal distribution for given x with given mean m and std s
    """
    return 1/(2*np.pi*s**2)*np.exp(-0.5*(x-m)**2/s**2)

def gmm_2comp(x, s1, s2, w1, w2):
    """
    True function is a mixture of two gaussians:
        w1*N(m1,s2) + w2*N(m2, s2)
    """
    # Getting back the objects:
    with open('means.pkl', 'rb') as f: 
        m1, m2 = pickle.load(f) 
        
    return w1*eval_norm(x, m1, s1) + w2*eval_norm(x, m2, s2)

def get_intersect(x_data, s1, s2, w1, w2):
    # Getting back the objects:
    with open('means.pkl', 'rb') as f: 
        m1, m2 = pickle.load(f) 
    
    g1 =  w1*eval_norm(x_data, m1, s1) 
    g2 =  w2*eval_norm(x_data, m2, s2) 
    
    R0 = np.argmin(abs(
            g1[int(0.1*len(x_data)) : int(0.9*len(x_data))] -\
            g2[int(0.1*len(x_data)) : int(0.9*len(x_data))])) +\
            int(0.1*len(x_data))
            
    x0 = x_data[R0]
    y0 =  w1*eval_norm(x0, m1, s1) 
            
    return g1, g2, x0, y0

def optimize_gmm_on_eye_aspect_ratios(aspect_ratios_v0, participant,report_results=False, plot_mixtures=True):
    """
    The intersection of the 2 components of the Gaussian Mixture Model is 
    considered to the threshold R0.
    """
    aspect_ratios = aspect_ratios_v0[participant]     
    
    # histogram
    plt.figure() 
    count, bins, patches = plt.hist(aspect_ratios, constants.NBINS_EYE_ASPECT_RATIO, density=1)
        
    # smooth histogram to find local maxima
    bin_mids = (bins[1:] + bins[:-1]) / 2 # midpoint of  bins
    window_len=11
    t_crop = int((window_len -1)/2)
    y = smooth(count, window_len, window='hanning')
    y_smooth = y[t_crop : -t_crop]


    m1_ind = np.min(np.where(np.sign(np.diff(y_smooth)) == -1))
    
    M0 = np.where(np.diff( np.sign(np.diff(y_smooth))) == 2)
    
    m1 = bin_mids[m1_ind] # first local maxima
    m2_ind = np.argmax(y_smooth) # global maxima
    m2 = bin_mids[m2_ind] # first local maxima
    
    plt.plot(bin_mids, y_smooth, 'k--')
    plt.plot(m1, y_smooth[m1_ind], 'o')
    plt.plot(m2, y_smooth[m2_ind], 'o')
    
    # Saving the objects:
    with open('means.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([m1, m2], f)
         
    
    aspect_ratios = np.asarray(aspect_ratios)
    aspect_ratios = aspect_ratios.reshape((-1, 1))

    x_data = bins[0:-1]
    y_data = count
    

    
    params, params_covariance = optimize.curve_fit(gmm_2comp, x_data, y_data,
                                               p0=[0.2,0.05, 0.7, 0.3])
    
    g1, g2, x0, y0 = get_intersect(x_data, params[0], params[1], params[2], params[3])
    plt.plot(x_data, g1)
    plt.plot(x_data, g2)
    plt.plot(x0, y0, 'o')
    
    xx0 = x_data[M0]
    yy0 = y_smooth[M0]
    plt.plot(xx0, yy0, '*')
    plt.title('Participant no: ' + participant)
    
    return x_data[M0], bin_mids, y_smooth


def detect():
    clipvar_fnames, aspect_ratios_v0, aspect_smooth, blink_thresholds = init_mats(illustrate = True)
    
    blink_thresholds = estimate_thresholds(clipvar_fnames, blink_thresholds, aspect_ratios_v0, aspect_smooth)
#    blink_thresholds['buse'] = 0.22
#    blink_thresholds['gokhan'] = 0.16
#    blink_thresholds['esra'] = 0.19
#    blink_thresholds['merve'] = 0.21
    set_thresholds_etc(clipvar_fnames, blink_thresholds)