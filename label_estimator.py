#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 16:33:01 2018

@author: zeynep
"""
import numpy as np
import csv


import file_tools as file_tools
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import constants
from importlib import reload
reload(constants)

class Label_estimator():
    """
    Blink detector pools aspect rations of individual and applies GMM on each 
    set. 
    """
    def __init__(self):
        
        self.neigh = KNeighborsClassifier(n_neighbors=12) #10~20 -> 10~40
        self.svc = SVC()

        
        self.clipvar_fnames = file_tools.get_data_fnames(constants.CLIP_VAR_DIR, 'clipvar_')  
        
        self.feats = []
        self.est_labels = []
        self.gt_labels = []
        self.gt_labels_binary = []

        for subject_no in range(0, constants.NSUBJECTS):    
            
            self.feats.append([])
            self.gt_labels.append([])
            self.gt_labels_binary.append([])
                        
            for clipvar_fname in self.clipvar_fnames:
                with open(str(clipvar_fname), 'rb') as f:
                    clip = pickle.load(f)
                    if ((int(clip['info']['participant']) is subject_no ) and\
                        clip['info']['valid'] ):
                        
                        if (\
                        (np.isnan(clip['dblink']) == False) and\
                        (np.isinf(clip['dblink']) == False) and\
                        (np.isnan(clip['ropen']) == False) and\
                        (np.isinf(clip['ropen']) == False) ):
                        
                             # nblink, aopen, aclosed, ropen, rclosed
                            self.feats[subject_no].append([
                                    clip['aopen'], \
                                    clip['dblink'],\
                                    clip['nblinks']\
                                    ])
                            self.gt_labels[subject_no].append([clip['info']['annotator2']])
                            self.gt_labels_binary[subject_no].append([np.sign(clip['info']['annotator2']-3)])
                        
 
            
    def estimate(self):
        self.classify()
        
        flat_gt = [item[0] for sublist in self.gt_labels for item in sublist]
        flat_est = [item for sublist in self.est_labels for item in sublist]
        
        with open('krip_infile.csv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(flat_gt, flat_est))
        
        
    def classify(self):
        """
        Runs the classifier for the given features
        
        """
        print('Classifying...')    
        
        for subject_no in range(0, constants.NSUBJECTS):  
            
            temp_est = []
            personal_feats = np.asarray(self.feats[subject_no])
            personal_gts = np.asarray(self.gt_labels[subject_no])
                        
            for n in range(len(personal_feats)):
                
                train_feats, train_gts, test_feat, test_gt = \
                self.split_train_test_feats(personal_feats, personal_gts, n)
                
                #self.neigh.fit(train_feats, np.ravel(train_gts))
                #est_label = self.neigh.predict([test_feat])
                
                self.svc.fit(train_feats, np.ravel(train_gts))
                est_label = self.svc.predict([test_feat])
                                
                temp_est.append(est_label[0])
               
            print('--------------------------------')
            print('Participant: {}'.format(subject_no))
            print('Var.\tAnnot-1\tAnnot-2'.format(subject_no))
            self.est_labels.append(temp_est)
            self.evaluate_indiv(subject_no)
                
        self.evaluate_cum()
                

          
    def split_train_test_feats(self, personal_feats, personal_gt, n):
        
        test_feat = personal_feats[n]
        test_gt = personal_gt[n]
        
        train_feats = []
        train_gts = []
        
        for i in range(len(personal_feats)):
            
            if i is not n:
                train_feats.append(personal_feats[i])
                train_gts.append(personal_gt[i])
                
        return train_feats, train_gts, test_feat, test_gt
    
    def evaluate_indiv(self, subject_no):
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
        for lab in self.gt_labels[subject_no]:
            gt_labels_array.append(lab[0])
        
        binary_error_perc = self.get_binary_error_indiv(subject_no, gt_labels_array)
        rmse = self.get_rmse_error_indiv(subject_no, gt_labels_array)
        mean_sagr = self.get_sagr_indiv(subject_no, gt_labels_array)
        #low_att_det_perf = self.det_low_attention(subject_no, gt_labels_array)
        
        print('Binary error (%)\t{:.2f}'.format(binary_error_perc))
        print('RMSE\t\t\t{:.2f}'.format(rmse))
        print('Mean SAGR\t\t{:.2f}'.format(mean_sagr))
        #print('Low att det perf(%)\t{:.2f}'.format(low_att_det_perf)) 
        
    def evaluate_cum(self):
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
        for subject_no in range(constants.NSUBJECTS):
            for lab in self.gt_labels[subject_no]:
                gt_labels_array.append(lab[0])
            
        binary_error_perc = self.get_binary_error_cum( gt_labels_array)
        rmse = self.get_rmse_error_cum( gt_labels_array)
        mean_sagr = self.get_sagr_cum( gt_labels_array)
        
        print('--------------------------------')
        print('Cumulative')
        print('Var.\tAnnot-1\tAnnot-2'.format(subject_no))
            
        print('Binary error (%)\t{:.2f}'.format(binary_error_perc))
        print('RMSE\t\t\t{:.2f}'.format(rmse))
        print('Mean SAGR\t\t{:.2f}'.format(mean_sagr))
    ##########################################################################
    def get_binary_error_indiv(self, subject_no, gt_labels_array): 
        binary_error = np.sum( np.invert ( np.equal(self.est_labels[subject_no] , gt_labels_array)))
        binary_error_perc = binary_error / len(self.est_labels[subject_no])
        return binary_error_perc
    
    def get_rmse_error_indiv(self, subject_no, gt_labels_array):
        
        
        rmse = np.sum(\
                      np.square(\
                                np.subtract(self.est_labels[subject_no] , gt_labels_array)))/\
                      len(self.est_labels[subject_no])
        return rmse

    def get_sagr_indiv(self, subject_no, gt_labels_array):
        
        if (np.max(self.est_labels[subject_no]) > 1) or\
        (np.min(self.est_labels[subject_no]) < -1):
            temp_est = [x - 3 for x in self.est_labels[subject_no]]
            temp_gt = [x - 3 for x in gt_labels_array]
        else:
            temp_est = self.est_labels[subject_no]
            temp_gt = gt_labels_array
        
        temp_est = np.sign(temp_est)
        temp_gt = np.sign(temp_gt)
        
        mean_sagr = np.sum(\
                           np.equal(self.est_labels[subject_no] , temp_gt))/\
                           len(self.est_labels[subject_no])
        return mean_sagr
    
    ##########################################################################
    def get_binary_error_cum(self, gt_labels_array): 
        
        temp_est = []
        for subject_no in range(constants.NSUBJECTS):
            temp_est.extend(self.est_labels[subject_no])
            
        binary_error = np.sum( np.invert ( np.equal(temp_est , gt_labels_array)))
        binary_error_perc = binary_error / len(temp_est)
        return binary_error_perc
    
    def get_rmse_error_cum(self, gt_labels_array):
        
        temp_est = []
        for subject_no in range(constants.NSUBJECTS):
            temp_est.extend(self.est_labels[subject_no])

        rmse = np.sum(\
                      np.square(\
                                np.subtract(temp_est , gt_labels_array)))/\
                      len(temp_est)
        return rmse

    def get_sagr_cum(self, gt_labels_array):
        
        temp_est2 = []
        for subject_no in range(constants.NSUBJECTS):
            temp_est2.extend(self.est_labels[subject_no])
            
        if (np.max(temp_est2) > 1) or\
        (np.min(temp_est2) < -1):
            temp_est = [x - 3 for x in temp_est2]
            temp_gt = [x - 3 for x in gt_labels_array]
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

    def det_low_attention(self, subject_no, gt_labels_array):
        
        if (np.max(self.est_labels[subject_no]) > 1) or\
        (np.min(self.est_labels[subject_no]) < -1):
            temp_est = [x - 3 for x in self.est_labels[subject_no]]
            temp_gt = [x - 3 for x in gt_labels_array]
        else:
            temp_est = self.est_labels[subject_no]
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
