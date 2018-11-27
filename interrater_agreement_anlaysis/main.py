#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:51:15 2018

@author: zeynep
"""
import file_tools as file
import reliability_measures as measures
import distributions


if __name__ == "__main__":

     fname = 'name_mapping_with_codes_num.csv'    
     data = file.readData(fname)
     
     fname = '../landmark_analysis_v2'

     
     alpha = measures.kriAlpha(data,'ordinal')
     kappa = measures.fleissKappa(data)
     
     binary_error_perc = measures.get_binary_error(data)
     rmse = measures.get_rmse_error(data)
     mean_sagr = measures.get_sagr(data)

     
     print('Krippendorf\'s alpha: \t %.2f' % alpha)
     print('Fleiss\' kappa: \t\t %.2f' % kappa)     
     print('Binary error: \t\t %.2f' % binary_error_perc)
     print('RMSE:\t\t\t %.2f' % rmse)
     print('Mean SAGR: \t\t %.2f' % mean_sagr)

     #timebin_vs_label_ann1 = distributions.get_field_vs_label(data, 'timebin', 'annotator1')
     #timebin_vs_label_ann2 = distributions.get_field_vs_label(data, 'timebin', 'annotator2')
     
     #exp_vs_label_ann1 = distributions.get_field_vs_label(data, 'experiment', 'annotator1')
     #exp_vs_label_ann2 = distributions.get_field_vs_label(data, 'experiment', 'annotator2')