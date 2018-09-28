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
import file_tools as file


import warnings
warnings.simplefilter('ignore', np.RankWarning)

from fractions import Fraction #add
from collections import Counter #ass



# number of landmarks on asingle frame
NLANDMARKS_FRAME = 68

       
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))



##########################################################################
#
# initialize figures
# 1. for displaying eye size
# 2. for displaying roll angle
#

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(0,0,640, 480)
plt.title('Eye size (px^2)') 

#fig3 = plt.figure()
#ax3 = fig3.add_subplot(111)
#mngr = plt.get_current_fig_manager()
#mngr.window.setGeometry(640,0,640, 480)
#plt.title('Roll angle (radians)') 

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(640,0,640, 480)
plt.title('blink') 




fig5 = plt.figure()
ax4 = fig5.add_subplot(111)
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(640,0,640, 480)
plt.title('histgram') 



##########################################################################

class Clip():
    """
    Clip is a collection of landmarks, head pose (acclereometer data), gaze data etc
    """
    def __init__(self, landmark_fname):
        self.video_fname = file.find_between_r(landmark_fname, '/datafiles/landmark/', '.dat.npy')
        self.landmark_sets = []
        self.nframes = 0
        self.eyesizes = {'momentary_right':[], 'momentary_left':[], 'momentary_mean':[],'mean_std_while_open':[]}
        self.poses = {'yaw':[], 'pitch': [], 'roll':[]}
        self.gazes = []
        self.n_blinks = 0
        self.dur_blinks = []
        self.info = {}
        
        
        
        self.T_r = {'r_eye':[]}
        self.T_l = {'l_eye':[]}
        self.T_m = {'m_eye':[]}
        
   
        self.T_r_all = {'r_eye_all':[]}

        
        

        
    
 
        
        
    def build_clip(self, exp_info, landmark_fname):
        """
        This function loads all the data elements for this clip:
        landmarks, head pose values, and eye gaze points
        
        Currently only the part about landmarks is written
        Later, I will add functions to load pose and gaze 
        """

        self.load_info(exp_info, landmark_fname)
        self.landmark_sets = np.load(landmark_fname)
        self.nframes = int(len(self.landmark_sets)/NLANDMARKS_FRAME)
        
        
    def load_info(self, exp_info, landmark_fname):
        """
        exp_info involves details about the experiments. Currently the data 
        fields are:
            
        video_fname, participant, experiment, clip_t0_unix, t0_min, timebin, 
        playlist, valid, annotator1, annotator2,
        
        I may change the file and add new fields.
        """
        ind = np.where(np.array(exp_info['video_fname']) == self.video_fname)
        
        
        #print(ind)
        
        # build a similar dict variable in this class with the same keys
        for key in exp_info.keys():
            self.info[key] = exp_info[key][ind[0][0]]
        
    def analyze(self):
        """
        This function will take the data elements and do the analysis on them
        For instance, it will take the landmarks and compute the number of blinks, size of the eyes etc
        Or it will take the pose angles and compute the mean pose etc
        Currently only the part relating the eyes is written
        """
        self.get_eyesizes_momentary()
        self.remove_blinks()
        # use one of the two function to get pose data
        #self.get_poses_LM()
        #self.get_poses_ACC(accmeter_fname)
        
        return self.T_r_all['r_eye_all']

        
    def get_eyesizes_momentary(self):
        """
        This function computes the mean size of the eyes (mean of right and left)
        for all frames.
        This means that the eyes may be closed or blinking etc.
        """
        # x and y coordinates of the landmarks (68 pts)
        # they are a bit redundant but make it easier to understand the code
        xpix = []
        ypix = []
        
        for i in range(self.nframes):      
     
            landmarks_frame = self.landmark_sets[i*NLANDMARKS_FRAME:(i+1)*NLANDMARKS_FRAME]
            xpix = landmarks_frame[:,0]
            ypix = 720-landmarks_frame[:,1] # dlib assumes that the origin is upper left corner
    
            size_r = PolyArea(xpix[36:42], ypix[36:42])
            size_l = PolyArea(xpix[42:48], ypix[42:48])
            size_m = (size_r + size_l)*0.5
        
            self.eyesizes['momentary_right'].append(size_r)
            self.eyesizes['momentary_left'].append(size_l)
            self.eyesizes['momentary_mean'].append(size_m)
            
        ############################################################
            
            p_36 = np.array(xpix[36],ypix[36])
            p_39 = np.array(xpix[39],ypix[39])    
            p_42 = np.array(xpix[42],ypix[42])
            p_45 = np.array(xpix[45],ypix[45])            
        
            self.dist_39_42 = np.linalg.norm(p_39 - p_42)
            self.dist_36_45 = np.linalg.norm(p_36 - p_45)
        

 #           print(self.video_fname, self.dist_39_42, self.dist_36_45)
        
 

                    
        ###########################################################
            
    
    def remove_blinks(self):
        # count number and duration of blinks
        T = np.median(self.eyesizes['momentary_mean']) - (np.max(self.eyesizes['momentary_mean']) - np.median(self.eyesizes['momentary_mean'])) 
        eyeclosed = [ sum( 1 for _ in group ) for key, group in itertools.groupby(self.eyesizes['momentary_mean'] < T ) if key ]
        
        # size of the eyes when they are open
        temp = np.array(self.eyesizes['momentary_mean'])
        eyesize_open =  temp[self.eyesizes['momentary_mean'] > T]
        self.eyesizes['mean_std_while_open']  = [np.mean(eyesize_open), np.std(eyesize_open)]  # mean and std  
        self.n_blinks = len(eyeclosed)
        self.dur_blinks = [np.mean(eyeclosed), np.std(eyeclosed)] # mean and std
        
        
        xpix = []
        ypix = []    
        
        
 #####
        
        s = 0 # count number of blink
        u = 0
        A_open = []
        nb = []
        X = []

#####        
        
        for i in range(self.nframes): 
            landmarks_frame = self.landmark_sets[i*NLANDMARKS_FRAME:(i+1)*NLANDMARKS_FRAME]
            xpix = landmarks_frame[:,0]
            ypix = 720-landmarks_frame[:,1] # dlib assumes that the origin is upper left corner
            
            
            p_36 = np.array(xpix[36],ypix[36])
            p_37 = np.array(xpix[37],ypix[37])
            p_38 = np.array(xpix[38],ypix[38])
            p_39 = np.array(xpix[39],ypix[39])         
            p_40 = np.array(xpix[40],ypix[40])
            p_41 = np.array(xpix[41],ypix[41])
            p_42 = np.array(xpix[42],ypix[42])
            p_43 = np.array(xpix[43],ypix[43])
            p_44 = np.array(xpix[44],ypix[44])
            p_45 = np.array(xpix[45],ypix[45])
            p_46 = np.array(xpix[46],ypix[46])
            p_47 = np.array(xpix[47],ypix[47])
        
            dist_37_41 = np.linalg.norm(p_37 - p_41)
            dist_38_40 = np.linalg.norm(p_38 - p_40)
        
            dist_43_47 = np.linalg.norm(p_43 - p_47)
            dist_44_46 = np.linalg.norm(p_44 - p_46)
                    
            dist_36_39 = np.linalg.norm(p_36 - p_39)
            dist_42_45 = np.linalg.norm(p_42 - p_45)
                
                
            #A = dist_37_41 + dist_38_40
            #B = 2 * dist_36_39
            #T__r = A / B
            
            
            C = dist_43_47 + dist_44_46
            D = 2 * dist_42_45
            T__l = C / D
            
            
            
            T__r = (np.linalg.norm(landmarks_frame[37] - landmarks_frame[41]) + \
                                    np.linalg.norm(landmarks_frame[38] - landmarks_frame[40]))\
                                    / 2 / np.linalg.norm(landmarks_frame[36] - landmarks_frame[39])    
                                    
            #print(T__r)
                      
            if T__r <= 0.229:
                s = s + 1
                #print(i, self.video_fname, T__r)
                #print(s)
               # print('BLINK')
            
            if T__r >= 0.229:
                A_open.append(T__r)
                u = u + 1
                #print(u)
                #print (self.video_fname,u,T__r)
                
            
            x = T__r - 0.229
            #print(i, self.video_fname, x)
            nb.append(x)
            #print(nb)
            #y = np.sign(nb)
            #print(y)
            #print(i, self.video_fname, y)
            #print(nb)
            #z = np.diff(y)
            #print(i, self.video_fname, z)
            


            
            
            
                
            self.T_r['r_eye'].append(T__r)
            self.T_l['l_eye'].append(T__l)
            
            
            

            
            


        
        ############################################################
            

                
            #print(i,self.video_fname,T__r)
            
                
                

          
            #T__m = ( T__r + T__l ) / 2
            
            #self.T_m['m_eye'].append(T__m)
            
            #print(i, self.video_fname, T__m)
            
            
            
        ratio = s / 300.0  
        #print(self.video_fname,s,ratio)
        #print[ratio,np.mean(A_open)]
  

        
        
        #print(self.video_fname,np.mean(A_open))
            
        
        self.T_r_all['r_eye_all'].extend( self.T_r['r_eye'] )
        #print(self.video_fname)
        #print(self.eyesizes['mean_std_while_open'])

        #print(nb)   
        x = np.sign(nb)
        #print(x)
        y = np.diff(x)
        #print(y) 
        z = Counter(y)
        #print(z)
        


    
    def get_poses_LM(self):
        """
        This function gets the head pose values from landmarks (LM)
        Basically it fits a line to the axis along the nose
        And computes roll angle as the slope of that line
        Currently it only computes roll angle
        Pitch and yaw angles are returned as 0
        """
          
        # x and y coordinates of the landmarks (68 pts)
        # they are a bit redundant but make it easier to understand the code
        xpix = []
        ypix = []
        
        #for landmark_set in self.landmark_sets[i*NLANDMARKS_FRAME:(i+1)*NLANDMARKS_FRAME]:      
        for i in range(self.nframes):      
                 
            landmark_set = self.landmark_sets[i*NLANDMARKS_FRAME:(i+1)*NLANDMARKS_FRAME]
            xpix = landmark_set[:,0]
            ypix = 720-landmark_set[:,1] # dlib assumes that the origin is upper left corner
    
            fit = np.polyfit(xpix[27:31], ypix[27:31],1)
            fit_fn = np.poly1d(fit) # for display purposes, maybe reduntand asometimes
            
            self.poses['roll'].append(np.arctan(fit[0]))
#            self.poses['pitch'].append(0)
#            self.poses['yaw'].append(0)
            
            #print(self.poses['roll'])
        M = np.median(self.poses['roll'])
        #print(self.video_fname)
#        print(M)




    def get_poses_ACC(self, accmeter_fname): # add
        """
        This function gets the head pose from the accelerometer data (ACC).
        It loads the ***_acc.npy and crops the relevant part 
        It computes the mean and std of yaw, pitch, roll
        
        This way we have an estimate for all 3 values.
        The estimation of roll angle is also supposed to be more reliable.
        
        The three columns should be organized as:
            Roll, Pitch, Yaw
        
        TODO
        """
        self.acc_fname = file.find_between_r(accmeter_fname, '/datafiles/accelerometer/', '_acc.npy')
        
        self.accmeter_sets = np.load(accmeter_fname)
        
        #print(self.acc_fname)
        
        #print(self.accmeter_sets)
        
        self.poses['roll'] = self.accmeter_sets[:,0]
        self.poses['pitch'] = self.accmeter_sets[:,1]
        self.poses['yaw'] = self.accmeter_sets[:,2]
        
        M = np.median(self.poses['yaw'])
        
        #print(self.video_fname) 
        #print(self.poses['roll'])
        #print(M)

            


          
    def illlustrate(self):
        """
        This function illustrates information derived from landmarks:
        1. Size of the eyes
        2. Roll angle of the head
        """
        # eye size
#        pl.figure(fig2.number)
#        plt.cla()
#        ax2.plot(self.eyesizes['momentary_mean'],'b.-')
#        ax2.grid(color='k', linestyle=':', linewidth=1)
#        plt.grid(True)
#        plt.show()
#        plt.title('Eye size (px^2) for '+self.video_fname) 
#        plt.draw()
           
        # roll angle throughout the video
#        pl.figure(fig3.number)
#        plt.cla()
#       ax3.plot(self.poses['roll'], 'b.-')
#       ax2.grid(color='k', linestyle=':', linewidth=1)
#        plt.grid(True)
#        plt.show()
#        plt.title('Roll angle (radians) for '+self.video_fname) 
#        plt.draw()
        
        
        
        # blink
#        pl.figure(fig4.number)
#        plt.cla()
#        ax4.plot(self.T_r['r_eye'],'b.-')
#        ax4.grid(color='k', linestyle=':', linewidth=1)
#        plt.grid(True)
#        plt.show()
#        plt.title('blink for '+self.video_fname) 
#        plt.draw()
        


        
        
#        pl.figure(fig5.number)
#        plt.cla()      
#        plt.hist(self.T_r_all['r_eye_all'], bins=50)
#        plt.title("Histogram")
#        plt.xlabel("x")
#        plt.ylabel("y")
#        plt.show()
        
                
        
        
        plt.pause(1)
        
        

        
