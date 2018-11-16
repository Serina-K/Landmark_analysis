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
     
     alpha = measures.kriAlpha(data,'ordinal')
     kappa = measures.fleissKappa(data)
     
     print('Krippendorf\'s alpha: \t %.2f' % alpha)
     print('Fleiss\' kappa: \t\t %.2f' % kappa)
     
     timebin_vs_label_ann1 = distributions.get_field_vs_label(data, 'timebin', 'annotator1')
     timebin_vs_label_ann2 = distributions.get_field_vs_label(data, 'timebin', 'annotator2')
     
     exp_vs_label_ann1 = distributions.get_field_vs_label(data, 'experiment', 'annotator1')
     exp_vs_label_ann2 = distributions.get_field_vs_label(data, 'experiment', 'annotator2')