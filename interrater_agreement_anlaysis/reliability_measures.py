#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 17:22:22 2018

@author: zeynep
"""

import math
import numpy as np

def kriAlpha(data,scale):
    
    """
    alpha = kriAlpha(data,scale)
    calculates Krippendorff's Alpha as a measure of inter-rater agreement
    
    data: rate matrix, each row is a rater or coder, each column is a case
    scale: level of measurement, supported are 'nominal', 'ordinal', 'interval'
    missing values have to be coded as #nan or inf
    
    For details about Krippendorff's Alpha see:
    Hayes, Andrew F. & Krippendorff, Klaus (2007). Answering the call for a
    standard reliability measure for coding data. Communication Methods and
    Measures, 1, 77-89
        
    Results for the two examples below have been verified:
        
    data = \
    [#nan   #nan   #nan   #nan   #nan     3     4     1     2     1     1     3     3   #nan     3; ...
    1   #nan     2     1     3     3     4     3   #nan   #nan   #nan   #nan   #nan   #nan   #nan; ...
    #nan   #nan     2     1     3     4     4   #nan     2     1     1     3     3   #nan     4];
        
    alpha nominal: 0.6914, ordinal: 0.8067, interval: 0.8108
        
    data = \
   [[1.1000,    2.1000,    5.0000,    1.1000,    2.0000], 
   [ 2.0000,    3.1000,    4.0000,    1.9000,    2.3000], 
   [1.5000,    2.9000,    4.5000,    4.4000,    2.1000], 
   [ math.nan,    2.6000 ,   4.3000,    1.1000,    2.3000]]
    
    alpha nominal: 0.0364, ordinal: 0.5482, interval: 0.5905
    
    """
    
    if len(data) > 2:
        temp1 = np.array([data['annotator1']])
        temp2 = np.array([data['annotator2']])
        data =  np.concatenate((temp1, temp2), 0)
        # get only those columns with 
        
        data = np.asanyarray(data)
    elif (len(data) == 2):
        temp1 = np.array([data['flat_gt']])
        temp2 = np.array([data['flat_est']])
        data =  np.concatenate((temp1, temp2), 0)

    

    allVals  =  np.unique(data)
    allVals  =  allVals[abs(allVals) < math.inf]

    # coincidence matrix
    coinMatr =  np.ones((len(allVals), len(allVals))) * float('nan')
    for r in range(0, len(allVals)):
        for c in range(r, len(allVals)):
            val = 0
            for d in range(0, len(data[0])):
                # find number of pairs
                thisEx = data[:,d]
                
                thisEx = thisEx[abs(thisEx) < math.inf]
                numEntr = len(thisEx)
                numP = 0
                for p1 in range(0, numEntr):
                    for p2 in range(0, numEntr):
                        if p1 == p2:
                            continue                        
                        if thisEx[p1] == allVals[r] and thisEx[p2] == allVals[c]:
                            numP = numP+1
                if numP:
                    val = val+numP/(numEntr-1)

            coinMatr[r,c] = val
            coinMatr[c,r] = val

    nc = np.sum(coinMatr,axis=1)
    n = np.sum(nc)

    # expected agreement
    expMatr = np.ones((len(allVals),len(allVals))) * float('nan')
    for i in range(0, len(allVals)):
        for j in range(0, len(allVals)):
            if i == j:
                val = nc[i]*(nc[j]-1)/(n-1);
            else:
                val = nc[i]*nc[j]/(n-1);

            expMatr[i,j] = val;


    # difference matrix
    diffMatr = np.zeros((len(allVals),len(allVals)))
    for i in range(0, len(allVals)):
        for j in range(i+1, len(allVals)):
            if i != j:
                if scale is 'nominal':
                    val = 1
                elif scale is 'ordinal':
                    val = np.sum(nc[i:j+1])-nc[i]/2-nc[j]/2
                    val = np.square(val)
                elif scale is 'interval':
                    val = (allVals[j]-allVals[i]) ** 2
                else:
                    print('unknown scale: ', scale)
            else:
                val = 0
            
            diffMatr[i,j] = val
            diffMatr[j,i] = val


    # observed - expected agreement
    mydo = 0
    de = 0
    for c  in range (0, len(allVals)):
        for k in range(c+1, len(allVals)):
            if scale is 'nominal':
                mydo = mydo+coinMatr[c,k]
                de = de+nc[c]*nc[k]
            elif scale is 'ordinal':
                mydo = mydo+coinMatr[c,k]*diffMatr[c,k]
                de = de+nc[c]*nc[k]*diffMatr[c,k]
            elif scale is 'interval':
                mydo = mydo+coinMatr[c,k]*(allVals[c]-allVals[k]) ** 2
                de = de+nc[c]*nc[k]*(allVals[c]-allVals[k]) ** 2
            else:
                print('unknown scale: ', scale)

    de = 1/(n-1)*de
    alpha = 1-mydo/de
    
    return alpha

def fleissKappa(data):
    """
    Just for trying this, 
    consider 4 people do binary labeling task for 10 samples:
        
    sample    tom brooks chris steve
    1           1      1     1     0
    2           0      0     0     0
    3           1      0     1     0
    4           0      0     0     0
    5           1      1     1     1
    6           0      1     1     1
    7           1      1     1     1
    8           1      1     1     1
    9           0      0     0     0
    10          1      1     0     0
    
    data =[[1, 1, 1, 0], [0,0,0,0,],[1,0,1,0,],[0,0,0,0], [1,1,1,1],[0,1,1,1],[1,1,1,1],[1,1,1,1],[0,0,0,0],[1,1,0,0]]
    data = np.transpose(data)
    
    This should return a kappa of 0.529.
    """
    
    temp1 = np.array([data['annotator1']])
    temp2 = np.array([data['annotator2']])
    data =  np.concatenate((temp1, temp2), 0)
        
    data = np.asanyarray(data)
    
    n_coders = len(data) # I know this
    n_subjects = len(data[0]) # number of labellings (clips)
    classes = np.unique(data) # possible labels
    
    mat = np.zeros((n_subjects, len(classes)))
    
    for i in range(0,n_subjects):
        for c, cc in enumerate(classes):
            mat[i][c] = np.sum(data[:,i]== cc) 
            
    p = mat.sum(axis = 0) / (n_coders*n_subjects)
    
    P = 1/n_coders/(n_coders-1) * (np.square(mat).sum(axis=1) - n_coders)
    P_tot = np.sum(P)
    
    # over the whole sheet
    P_bar = P_tot / n_subjects;
    P_e = np.square(p).sum();

    kappa = (P_bar - P_e) / (1-P_e);
    
    return kappa

def get_binary_error(data): 
    
    temp1 = np.array([data['annotator1']])
    temp2 = np.array([data['annotator2']])
    
    binary_error = np.sum( np.invert ( np.equal(temp1 , temp2)))
    binary_error_perc = binary_error / len(temp1[0])
    return binary_error_perc

def get_sagr(data):
    
    temp1 = np.array([data['annotator1']])
    temp2 = np.array([data['annotator2']])
    
    if (np.max(temp1) > 1) or (np.min(temp1) < -1):
        temp_est = [x - 3 for x in temp1]
        temp_gt = [x - 3 for x in temp2]
    else:
        temp_est = temp1
        temp_gt = temp2
    
    temp_est = np.sign(temp_est)
    temp_gt = np.sign(temp_gt)
    
    mean_sagr = np.sum(\
                       np.equal(temp1 , temp2))/\
                       len(temp1[0])
    return mean_sagr

def get_rmse_error(data):
    
        
    temp1 = np.array([data['annotator1']])
    temp2 = np.array([data['annotator2']])
    
    rmse = np.sum(\
                  np.square(\
                            np.subtract(temp1 , temp2)))/\
                  len(temp1[0])
    return rmse