#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 16:33:01 2018

@author: zeynep
"""
import numpy as np
import csv
import copy

import file_tools as file_tools
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from importlib import reload

import constants
reload(constants)

import preferences
reload(preferences)

def init_classifier():
    neigh = KNeighborsClassifier(n_neighbors= preferences.N_NEIGHBORS) #10~20 -> 10~40
    svc = SVC()

    return neigh, svc
    
def init_conf_mat(gt_labels):
    conf_mat = {}
    ulabels = []
    
    temp = list(gt_labels.values())
    flat_list = [item for sublist in temp for item in sublist]
    ulabels = np.unique(flat_list)
    
    # get unique values here
    
    for m in ulabels:
        conf_mat[str(m)] = {}
        for n in ulabels:
            conf_mat[str(m)][str(n)] = 0
    
    return conf_mat, ulabels

def init_mats():
    
    clipvar_fnames = file_tools.get_data_fnames(constants.CLIP_VAR_DIR, 'clipvar_')  
    
    feats = {}
    est_labels = {}
    gt_labels = {}
    gt_labels_binary = {}
    
    for participant in preferences.SUBJECTS:
        feats[participant] = []
        est_labels[participant] = []
        gt_labels[participant] = []
        gt_labels_binary[participant] = []
            
    for clipvar_fname in clipvar_fnames:
        with open(str(clipvar_fname), 'rb') as f:
            clip = pickle.load(f)
        for participant in preferences.SUBJECTS:
                if (clip['info']['participant'] == participant  and\
                    clip['info']['experiment'] in preferences.EXPERIMENTS and\
                    clip['info']['valid'] ):
                    
                    temp = []
                    for f in preferences.FEATS:
                        temp.append( clip[f])
                    
                    if ( np.sum( np.isnan(temp)) == 0) :
                        feats[participant].append(temp)
                        gt_labels[participant].append([clip['info'][preferences.GT_ANNOTATOR]])
                        gt_labels_binary[participant].append([np.sign(clip['info'][preferences.GT_ANNOTATOR]-3)])
             
    return feats, est_labels, gt_labels, gt_labels_binary
   

    
    
def classify(neigh, svc, feats, gt_labels, est_labels, conf_mat, ulabels):
    """
    Runs the classifier for the given features
    
    """
    print('Classifying...')    
    
    for participant in preferences.SUBJECTS:
        
        temp_est = []
        personal_feats = np.asarray(feats[participant])
        personal_gts = np.asarray(gt_labels[participant])
                    
        for n in range(len(personal_feats)):
            
            train_feats, train_gts, test_feat, test_gt = \
            split_train_test_feats(personal_feats, personal_gts, n)
            
            if preferences.CLASSIFIER is 'knn':
                neigh.fit(train_feats, np.ravel(train_gts))
                est_label = neigh.predict([test_feat])
            elif preferences.CLASSIFIER is 'svc':
                svc.fit(train_feats, np.ravel(train_gts))
                est_label = svc.predict([test_feat])
                            
            temp_est.append(est_label[0])
            
            conf_mat[str(test_gt[0])][str(est_label[0])] += 1
            

        est_labels[participant] = temp_est.copy()
        if len(est_labels[participant]):
            evaluate_indiv(participant, est_labels[participant], gt_labels[participant])
        else:
            print('Nothing to evaluete for  {}'.format(participant))
       
    conf_mat_norm = copy.deepcopy(conf_mat)
    for m in ulabels:
        row_sum  = 0
        for n in ulabels:
            row_sum += conf_mat_norm[str(m)][str(n)]
        for n in ulabels:
            conf_mat_norm[str(m)][str(n)] = conf_mat_norm[str(m)][str(n)]   / row_sum         
        
    evaluate_cum(conf_mat, conf_mat_norm, gt_labels, est_labels, ulabels)
            

      
def split_train_test_feats(personal_feats, personal_gt, n):
    
    test_feat = personal_feats[n]
    test_gt = personal_gt[n]
    
    train_feats = []
    train_gts = []
    
    for i in range(len(personal_feats)):
        
        if i is not n:
            train_feats.append(personal_feats[i])
            train_gts.append(personal_gt[i])
            
    return train_feats, train_gts, test_feat, test_gt

def evaluate_indiv(participant, pest_labels, pgt_labels):
    """
    Thus function computes error in terms of various measures:
        
        Binary error: 1 for successful prediction of label (either for 
        labels 1-5 or -1/0/1)
        
        RMSE error: Root mean suare of the error (either for labels 1-5 or 
        -1/0/1)
        
        SAGR: standart signed agreement metric
        
        Det_low_attention: is the same as SAGR but cares only abou 
        detection of low attettion (4,5 or -1, depeding on the coding)
    """
    
    gt_labels_array = []
    for lab in pgt_labels:
        gt_labels_array.append(lab[0])
    
    binary_error_perc = get_binary_error_indiv(pest_labels, gt_labels_array)
    rmse = get_rmse_error_indiv(pest_labels, gt_labels_array)
    mean_sagr = get_sagr_indiv(pest_labels, gt_labels_array)
    #low_att_det_perf = det_low_attention(subject_no, gt_labels_array)
    
#    print('--------------------------------')
#    print('Participant: {}'.format(participant))
#        
#    print('Binary error (%)\t{:.2f}'.format(binary_error_perc))
#    print('RMSE\t\t\t{:.2f}'.format(rmse))
#    print('Mean SAGR\t\t{:.2f}'.format(mean_sagr))
    #print('Low att det perf(%)\t{:.2f}'.format(low_att_det_perf)) 
    
def evaluate_cum(conf_mat, conf_mat_norm, gt_labels, est_labels, ulabels):
    """
    Thus function computes error in terms of various measures:
        
        Binary error: 1 for successful prediction of label (either for 
        labels 1-5 or -1/0/1)
        
        RMSE error: Root mean suare of the error (either for labels 1-5 or 
        -1/0/1)
        
        SAGR: standart signed agreement metric
        
        Det_low_attention: is the same as SAGR but cares only abou 
        detection of low attettion (4,5 or -1, depeding on the coding)
    """
    gt_labels_array = []
    for participant in preferences.SUBJECTS:
        for lab in gt_labels[participant]:
            gt_labels_array.append(lab[0])
        
    binary_error_perc = get_binary_error_cum( est_labels, gt_labels_array)
    rmse = get_rmse_error_cum(est_labels, gt_labels_array)
    mean_sagr = get_sagr_cum(est_labels, gt_labels_array)
    
    print('--------------------------------')
    print('Cumulative')
    

    for m in ulabels:
        print('{}\t'.format(m), end='', flush=True)
        for n in ulabels:
            print('{:.0f}\t'.format(conf_mat[str(m)][str(n)]), end='', flush=True)
        print('\n')
    print('\n')    
    for m in ulabels:
        print('{}\t'.format(m), end='', flush=True)
        for n in ulabels:
            print('{:.2f}\t'.format(conf_mat_norm[str(m)][str(n)]), end='', flush=True)
        print('\n')
            
        
    print('Var.\tAnnot-1\tAnnot-2'.format(participant))
        
    print('Binary error (%)\t{:.2f}'.format(binary_error_perc))
    print('RMSE\t\t\t{:.2f}'.format(rmse))
    print('Mean SAGR\t\t{:.2f}'.format(mean_sagr))
##########################################################################
def get_binary_error_indiv(pest_labels, gt_labels_array): 
    # pest labels are the labels estimated for single participant
    binary_error = np.sum( np.invert ( np.equal(pest_labels , gt_labels_array)))
    binary_error_perc = binary_error / len(pest_labels)
    return binary_error_perc

def get_rmse_error_indiv(pest_labels, gt_labels_array):
    # pest labels are the labels estimated for single participant
  
    
    rmse = np.sum(\
                  np.square(\
                            np.subtract(pest_labels , gt_labels_array)))/\
                  len(pest_labels)
    return rmse

def get_sagr_indiv(pest_labels, gt_labels_array):
    
    if (np.max(pest_labels) > 1) or\
    (np.min(pest_labels) < -1):
        temp_est = [x - 3.5 for x in pest_labels]
        temp_gt = [x - 3.5 for x in gt_labels_array]
    else:
        temp_est = pest_labels
        temp_gt = gt_labels_array
    
    temp_est = np.sign(temp_est)
    temp_gt = np.sign(temp_gt)
    
    mean_sagr = np.sum(\
                       np.equal(temp_est , temp_gt))/\
                       len(pest_labels)
    return mean_sagr

##########################################################################
def get_binary_error_cum(est_labels, gt_labels_array): 
    
    temp_est = []
    for participant in preferences.SUBJECTS:
        temp_est.extend(est_labels[participant])
        
    binary_error = np.sum( np.invert ( np.equal(temp_est , gt_labels_array)))
    binary_error_perc = binary_error / len(temp_est)
    return binary_error_perc

def get_rmse_error_cum(est_labels, gt_labels_array):
    
    temp_est = []
    for participant in preferences.SUBJECTS:
        temp_est.extend(est_labels[participant])

    rmse = np.sum(\
                  np.square(\
                            np.subtract(temp_est , gt_labels_array)))/\
                  len(temp_est)
    return rmse

def get_sagr_cum(est_labels, gt_labels_array):
    
    temp_est2 = []
    for participant in preferences.SUBJECTS:
        temp_est2.extend(est_labels[participant])
        
    if (np.max(temp_est2) > 1) or\
    (np.min(temp_est2) < -1):
        temp_est = [x - 3.5 for x in temp_est2]
        temp_gt = [x - 3.5 for x in gt_labels_array]
    else:
        temp_est = temp_est2
        temp_gt = gt_labels_array
    
    temp_est = np.sign(temp_est)
    temp_gt = np.sign(temp_gt)
    
    mean_sagr = np.sum(\
                       np.equal(temp_est , temp_gt))/\
                       len(temp_est)
    return mean_sagr    
##########################################################################

def det_low_attention(participant, est_labels, gt_labels_array):
    
    if (np.max(est_labels[participant]) > 1) or\
    (np.min(est_labels[participant]) < -1):
        temp_est = [x - 3 for x in est_labels[participant]]
        temp_gt = [x - 3 for x in gt_labels_array]
    else:
        temp_est = est_labels[participant]
        temp_gt = gt_labels_array
    
    # 1 is very focused and 5 is completely out of focus

    ind_low_att = np.where(np.sign(temp_gt) == 1)[0]
    
    p = 0
    for i in ind_low_att:
        if temp_est[i] == 1:
            p += temp_est[i]
        
    if len(ind_low_att) > 0:
        return p/len(ind_low_att)
    else:
        return 1
    
    
        
def estimate():
    
    feats, est_labels, gt_labels, gt_labels_binary = init_mats()
    neigh, svc = init_classifier()
    conf_mat, ulabels  = init_conf_mat(gt_labels)
    
    classify(neigh, svc, feats, gt_labels, est_labels, conf_mat, ulabels)
    
    flat_gt = [item[0] for sublist in gt_labels for item in sublist]
    flat_est = [item for sublist in est_labels for item in sublist]
    
    with open('krip_infile.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(flat_gt, flat_est))    