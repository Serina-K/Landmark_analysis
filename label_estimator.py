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


from importlib import reload

import constants
reload(constants)

import preferences
reload(preferences)

class Label_estimator():
    """
    Blink detector pools aspect rations of individual and applies GMM on each 
    set. 
    """
    def __init__(self):
        
        self.neigh = KNeighborsClassifier(n_neighbors= preferences.N_NEIGHBORS) #10~20 -> 10~40
        self.svc = SVC()

        
        self.clipvar_fnames = file_tools.get_data_fnames(constants.CLIP_VAR_DIR, 'clipvar_')  
        
        self.feats = {}
        self.est_labels = {}
        self.gt_labels = {}
        self.gt_labels_binary = {}
        
        for participant in preferences.SUBJECTS:
            self.feats[participant] = []
            self.est_labels[participant] = []
            self.gt_labels[participant] = []
            self.gt_labels_binary[participant] = []
            

        
        self.conf_mat = {}

        
        for clipvar_fname in self.clipvar_fnames:
            with open(str(clipvar_fname), 'rb') as f:
                clip = pickle.load(f)
            for participant in preferences.SUBJECTS:
                    if (clip['info']['participant'] == participant  and\
                        clip['info']['experiment'] in preferences.EXPERIMENTS and\
                        clip['info']['valid'] ):
                        
                        if (\
                        (np.isnan(clip['dblink_m']) == False) and\
                        (np.isinf(clip['dblink_m']) == False) and\
                        (np.isnan(clip['r_open']) == False) and\
                        (np.isinf(clip['r_open']) == False) ):
                        
                             # nblinks, aopennorm, aclosed, ropen, rclosed
                            self.feats[participant].append([
                                    clip['aopennorm'], \
                                    clip['dblink_m'] / constants.NFRAMES,\
                                    clip['r_open']
                                    ])
                            self.gt_labels[participant].append([clip['info'][preferences.GT_ANNOTATOR]])
                            self.gt_labels_binary[participant].append([np.sign(clip['info'][preferences.GT_ANNOTATOR]-3)])
                 
        self.ulabels = []
        self.init_conf_mat()
        
        
    def init_conf_mat(self):
        temp = list(self.gt_labels.values())
        flat_list = [item for sublist in temp for item in sublist]
        self.ulabels = np.unique(flat_list)
        
        # get unique values here
        
        for m in self.ulabels:
            self.conf_mat[str(m)] = {}
            for n in self.ulabels:
                self.conf_mat[str(m)][str(n)] = 0
        
            
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
        
        for participant in preferences.SUBJECTS:
            
            temp_est = []
            personal_feats = np.asarray(self.feats[participant])
            personal_gts = np.asarray(self.gt_labels[participant])
                        
            for n in range(len(personal_feats)):
                
                train_feats, train_gts, test_feat, test_gt = \
                self.split_train_test_feats(personal_feats, personal_gts, n)
                
                if preferences.CLASSIFIER is 'knn':
                    self.neigh.fit(train_feats, np.ravel(train_gts))
                    est_label = self.neigh.predict([test_feat])
                elif preferences.CLASSIFIER is 'svc':
                    self.svc.fit(train_feats, np.ravel(train_gts))
                    est_label = self.svc.predict([test_feat])
                                
                temp_est.append(est_label[0])
                
                self.conf_mat[str(test_gt[0])][str(est_label[0])] += 1
                
            print('--------------------------------')
            print('Participant: {}'.format(participant))
            print('Var.\tAnnot-1\tAnnot-2'.format(participant))
            self.est_labels[participant] = temp_est.copy()
            self.evaluate_indiv(participant)
                
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
    
    def evaluate_indiv(self, participant):
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
        for lab in self.gt_labels[participant]:
            gt_labels_array.append(lab[0])
        
        binary_error_perc = self.get_binary_error_indiv(participant, gt_labels_array)
        rmse = self.get_rmse_error_indiv(participant, gt_labels_array)
        mean_sagr = self.get_sagr_indiv(participant, gt_labels_array)
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
        for participant in preferences.SUBJECTS:
            for lab in self.gt_labels[participant]:
                gt_labels_array.append(lab[0])
            
        binary_error_perc = self.get_binary_error_cum( gt_labels_array)
        rmse = self.get_rmse_error_cum( gt_labels_array)
        mean_sagr = self.get_sagr_cum( gt_labels_array)
        
        print('--------------------------------')
        print('Cumulative')
        
        for m in self.ulabels:
            for n in self.ulabels:
                print('{}\t'.format(self.conf_mat[str(m)][str(n)]), end='', flush=True)
            print('\n')
                
            
        print('Var.\tAnnot-1\tAnnot-2'.format(participant))
            
        print('Binary error (%)\t{:.2f}'.format(binary_error_perc))
        print('RMSE\t\t\t{:.2f}'.format(rmse))
        print('Mean SAGR\t\t{:.2f}'.format(mean_sagr))
    ##########################################################################
    def get_binary_error_indiv(self, participant, gt_labels_array): 
        binary_error = np.sum( np.invert ( np.equal(self.est_labels[participant] , gt_labels_array)))
        binary_error_perc = binary_error / len(self.est_labels[participant])
        return binary_error_perc
    
    def get_rmse_error_indiv(self, participant, gt_labels_array):
        
        
        rmse = np.sum(\
                      np.square(\
                                np.subtract(self.est_labels[participant] , gt_labels_array)))/\
                      len(self.est_labels[participant])
        return rmse

    def get_sagr_indiv(self, participant, gt_labels_array):
        
        if (np.max(self.est_labels[participant]) > 1) or\
        (np.min(self.est_labels[participant]) < -1):
            temp_est = [x - 3 for x in self.est_labels[participant]]
            temp_gt = [x - 3 for x in gt_labels_array]
        else:
            temp_est = self.est_labels[participant]
            temp_gt = gt_labels_array
        
        temp_est = np.sign(temp_est)
        temp_gt = np.sign(temp_gt)
        
        mean_sagr = np.sum(\
                           np.equal(self.est_labels[participant] , temp_gt))/\
                           len(self.est_labels[participant])
        return mean_sagr
    
    ##########################################################################
    def get_binary_error_cum(self, gt_labels_array): 
        
        temp_est = []
        for participant in preferences.SUBJECTS:
            temp_est.extend(self.est_labels[participant])
            
        binary_error = np.sum( np.invert ( np.equal(temp_est , gt_labels_array)))
        binary_error_perc = binary_error / len(temp_est)
        return binary_error_perc
    
    def get_rmse_error_cum(self, gt_labels_array):
        
        temp_est = []
        for participant in preferences.SUBJECTS:
            temp_est.extend(self.est_labels[participant])

        rmse = np.sum(\
                      np.square(\
                                np.subtract(temp_est , gt_labels_array)))/\
                      len(temp_est)
        return rmse

    def get_sagr_cum(self, gt_labels_array):
        
        temp_est2 = []
        for participant in preferences.SUBJECTS:
            temp_est2.extend(self.est_labels[participant])
            
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

    def det_low_attention(self, participant, gt_labels_array):
        
        if (np.max(self.est_labels[participant]) > 1) or\
        (np.min(self.est_labels[participant]) < -1):
            temp_est = [x - 3 for x in self.est_labels[participant]]
            temp_gt = [x - 3 for x in gt_labels_array]
        else:
            temp_est = self.est_labels[participant]
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