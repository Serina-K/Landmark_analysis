#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:32:25 2018

@author: zeynep
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import itertools
from os import listdir


import sys
from PyQt4 import QtGui
from PyQt4 import QtCore



import time

nlandmarks_frame = 68

       
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

datafile_dir = './clips/' # change this to your directory
data_fnames = [datafile_dir + f for f in listdir(datafile_dir) if 'alsint.dat.npy' in f] ## add adesso.

probes = [dict() for x in range(len(data_fnames))]


n_blinks = np.zeros((len(data_fnames),1))
dur_blinks = np.zeros((len(data_fnames),2)) # 2 cols for mean and std

##########################################################################
#
# initialize figures
# 1. for displaying landmarks 
# 2. for displaying eye size
# 3. for displaying roll angle
#
#fig1, ax = plt.subplots()
#mngr = plt.get_current_fig_manager()
#mngr.window.setGeometry(0,0,1280, 1080) # to put it into the upper left corner 
#plt.show()
#axes = plt.gca()
#axes.set_xlim(0, 1280)
#axes.set_ylim(0, 720)
#line, = axes.plot(0, 0, 'ro')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
mngr = plt.get_current_fig_manager()
#mngr.window.setGeometry(0,0,640, 480)
plt.title('Eye size (px^2)') 

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
mngr = plt.get_current_fig_manager()
#mngr.window.setGeometry(640,0,640, 480)
plt.title('Roll angle (radians)') 

##########################################################################


# process each data file
for f, fname in enumerate(data_fnames):
    landmarks_video = np.load(fname)
    nframes = int(len(landmarks_video)/nlandmarks_frame)

    eyesize = np.zeros((nframes, 3)) # initialize eyesize array 
    roll = np.zeros((nframes, 1)) # output array roll angle array 
      
    # x and y coordinates of the landmarks (68 pts)
    # they are a bit redundant but make it easier to understand the code
    xpix = []
    ypix = []

    #for i in range(10): # running the code for just a few frames
    for i in range(nframes):      
 
        landmarks_frame = landmarks_video[i*nlandmarks_frame:(i+1)*nlandmarks_frame]
        xpix = landmarks_frame[:,0]
        ypix = 720-landmarks_frame[:,1] #    dlib assumes that the origin is upper left corner

        ##########################################################################
        #
        # get size of each eye and mean eye size
        #
        eyesize_right = PolyArea(xpix[36:42], ypix[36:42])
        eyesize_left = PolyArea(xpix[42:48], ypix[42:48])
        eyesize_mean = (eyesize_right + eyesize_left)*0.5
    
        eyesize[i] = [eyesize_right, eyesize_left, eyesize_mean]
        
        
        
        
 #       a = np.array(xpix[39],ypix[39])
 #       b = np.array(xpix[42],ypix[42])
 #       dis = a - b
 #       np.linalg.norm(dis)
        
        
        short = np.linalg.norm(landmarks_frame[39] - landmarks_frame[42])
        longer = np.linalg.norm(landmarks_frame[36] - landmarks_frame[45])
        
        print(short, longer)
        


        ##########################################################################
        #
        # get roll angle
        #
        fit = np.polyfit(xpix[27:31], ypix[27:31],1)
        fit_fn = np.poly1d(fit) 
        roll[i] = np.arctan(fit[0])
    
    
        #    ##########################################################################
        #    #
        #    # for illustrating landmarks and annotating each landmark
        #    # uncommment the part before the loop as well
        #    #
        #    ##########################################################################
        #    line.set_xdata(xpix)
        #    line.set_ydata(ypix)
        #    line.set_xdata(xpix[27:31])
        #    line.set_ydata(ypix[27:31])
        #    plt.plot(xpix[27:31], fit_fn(xpix[27:31]), '--k') # for displaying the line on the nose
        #
        #     
        ##    for j in range(0,nlandmarks_frame):
        #        ann = ax.annotate(str(j), (xpix[j],ypix[j]),fontsize=5)
        #    
        #    plt.draw()
        #    plt.pause(1e-17)
        #    time.sleep(0.1)
          
    # count number and duration of blinks
    T = np.median(eyesize[:,2]) - (np.max(eyesize[:,2]) - np.median(eyesize[:,2])) 
    eyeclosed = [ sum( 1 for _ in group ) for key, group in itertools.groupby(  eyesize[:,2] < T ) if key ]
    n_blinks = len(eyeclosed)
    
    eyesize_open = eyesize[eyesize[:,2] > T, 2] 

    # put into the output variable
    probes[f]['fname'] = fname    
    probes[f]['n_blinks'] = n_blinks
    probes[f]['dur_blinks'] = [np.mean(eyeclosed), np.std(eyeclosed)] # mean and std
    probes[f]['eyesize'] = [np.mean(eyesize_open), np.std(eyesize_open)]  # mean and std  
    probes[f]['roll'] = [np.mean(roll), np.std(roll)]  # mean and std  

   
    ###########################################################################
    #
    # eye size throughout the video
    #
    ###########################################################################  
#    pl.figure(fig2.number)
#    plt.cla()
#    ax2.plot(eyesize[:,2],'b.-')
#    ax2.grid(color='k', linestyle=':', linewidth=1)
#    plt.grid(True)
#    plt.show()
#    plt.title('Eye size (px^2) for'+fname) 
#    plt.draw()


    ###########################################################################
    #
    # roll angle throughout the video
    #
    ###########################################################################  
#    pl.figure(fig3.number)
#    plt.cla()
#    ax3.plot(roll, 'b.-')
#    ax2.grid(color='k', linestyle=':', linewidth=1)
#    plt.grid(True)
#    plt.show()
#    plt.title('Roll angle (radians)for'+fname) 
#    plt.pause(1)
#    plt.draw()

#np.save('probes2.npy', probes) 