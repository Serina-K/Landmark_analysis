#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:57:53 2018

@author: zeynep
"""
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

import constants
from importlib import reload
reload(constants)

###############################################################################
#
# Trivial maths
#
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def limit_range(angle):
    """
    This function limits the range of the in[put angle to [-pi/2, pi/2].
    
    Namely, the input is one hed pose value (usually roll angle). It is derived 
    from the pose of the nose, and the range is [-pi, pi]. But actually the head 
    (or the nose) does not rotate 2pi. It may be anywhere fully horizontal from 
    left to right, ie range is pi (from 0 to pi).

    """
    if -np.pi < angle and angle < -1/2*np.pi:
        return angle + np.pi
    elif -1/2*np.pi < angle and angle < 0:
        return angle + np.pi
    else :
        return angle


###############################################################################
#
# Display stuff
#
    

def prep_figures():
    #
    # initialize figures
    # 1. for displaying eye size
    # 2. for displaying roll angle
    #
    plt.close("all")
    
#    plt.figure(1)
#    mngr = plt.get_current_fig_manager()
#    mngr.window.setGeometry(0,0,320, 240)
#    plt.title('Eye size (px^2)') 
    
    fig3 = plt.figure(2)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(640,0,640, 480)
    plt.title('Head pose') 
    
#    plt.figure(3)
#    mngr = plt.get_current_fig_manager()
#    mngr.window.setGeometry(400,0,320, 240)
#    plt.title('Blink') 
#    
#    plt.figure(4)
#    mngr = plt.get_current_fig_manager()
#    mngr.window.setGeometry(750,0,320, 300)
#    plt.title('Histogram') 
    
def pool_angles(clip):
    """
    Pose values are stored inside each frame. In order to pool all poses 
    concering a clip, I use this pooling function
    """
    pose_LM = {'yaw':[], 'pitch': [], 'roll':[]} # pose from landmarks
    pose_ACC = {'yaw':[], 'pitch': [], 'roll':[]} # pose from landmarks
    
    print('video_fname {} no frames {}'.format(clip.video_fname, len(clip.frames)) )
    
    for frame in clip.frames:
        
        pose_LM['roll'].append(frame.pose_LM['roll'])
        
        if frame.pose_ACC['yaw'] is not 0: 
            pose_ACC['yaw'].append(frame.pose_ACC['yaw'])
            pose_ACC['roll'].append(frame.pose_ACC['roll'])
            pose_ACC['pitch'].append(frame.pose_ACC['pitch'])

        
#        if isinstance(frame.pose_ACC['yaw'], (int)):
#          if frame.pose_ACC['yaw'] is not 0: 
#              pose_ACC['yaw'].append(frame.pose_ACC['yaw'])
#              pose_ACC['roll'].append(frame.pose_ACC['roll'])
#              pose_ACC['pitch'].append(frame.pose_ACC['pitch'])
#        else:
#            if len(frame.pose_ACC['yaw']) > 1: 
#                pose_ACC['yaw'].extend(frame.pose_ACC['yaw'])
#                pose_ACC['roll'].extend(frame.pose_ACC['roll'])
#                pose_ACC['pitch'].extend(frame.pose_ACC['pitch'])            

        
    return pose_LM, pose_ACC

def illustrate(clip):
    """
    This function illustrates information derived from landmarks:
    1. Size of the eyes
    2. Roll angle of the head
    """
    if clip.info['valid']:
        
        pose_LM, pose_ACC = pool_angles(clip)
    #    # eye size
    #    fig = pl.figure(1) # pl.figure(fig2.number)
    #    plt.cla()
    #    ax = fig.axes[0]
    #    ax.plot(clip.frame['eye_size']['mean'],'b.-')
    #    ax.grid(color='k', linestyle=':', linewidth=1)
    #    plt.grid(True)
    #    plt.show()
    #    plt.title('Eye size (px^2) for '+clip.video_fname) 
    #    plt.draw()
       
    #   #  roll angle throughout the video
    #    pl.figure(fig3.number)
    #    plt.cla()
    #    ax3.plot(clip.poses['roll'], 'b.-')
    #    ax2.grid(color='k', linestyle=':', linewidth=1)
    #    plt.grid(True)
    #    plt.show()
    #    plt.title('Roll angle (radians) for '+self.video_fname) 
    #    plt.draw()
    
        #  head pose
        fig = pl.figure(3)
        plt.cla()
        ax = fig.add_subplot(111)
        axes = plt.gca()
        axes.set_ylim([-np.pi,np.pi])
        plt1 = ax.plot(pose_LM['roll'] ,'b.-', label='LM roll')
        plt2 = ax.plot(pose_ACC['yaw'] ,'r.-', label='ACC yaw')
        plt3 = ax.plot(pose_ACC['roll'] ,'m.-', label='ACC roll')
        plt4 = ax.plot(pose_ACC['pitch'] ,'k.-', label='ACC pitch')
        
        plt.legend()
        
        ax.grid(color='k', linestyle=':', linewidth=1)
        plt.grid(True)
        plt.show()
        plt.title('blink for ' + clip.video_fname) 
        plt.draw()
    
    #    fig = pl.figure(4)
    #    plt.cla()      
    #    plt.hist(clip.T_r_all['r_eye_all'], bins=50)
    #    plt.title("Histogram")
    #    plt.xlabel("x")
    #    plt.ylabel("y")
    #    plt.show()
        
        plt.pause(1)
        
        
    
        