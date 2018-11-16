#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:01:37 2018

@author: zeynep
"""
import numpy as np
import matplotlib.pyplot as plt

def get_field_vs_label(data, data_field, annotator_no):
    
    fields = np.unique(data[data_field])
    labels = np.unique(data[annotator_no])
    
    field_vs_label = np.zeros((len(fields), len(labels)))
    for fi, f in enumerate(fields):
        for li, l in enumerate(labels):
            field_vs_label[fi, li] = sum(data[annotator_no][data[data_field] == f] == l)
         
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    
    bottom = [0 for x in fields] 
    for li, l in enumerate(labels):
        ax1.bar(fields, field_vs_label[:,li], 0.3, bottom )
        bottom += field_vs_label[:,li]
            
    
    plt.ylabel('Number of labels')
    plt.xlabel(data_field)

    plt.yticks(np.arange(0, int(np.max(bottom)*1.2), 10))

    plt.show()
    
    return field_vs_label
    
