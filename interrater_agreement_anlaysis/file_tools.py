#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:11:04 2018

@author: zeynep
"""
import numpy.ma as MA 
import numpy as np

def readHeader(fname): 
    # Open the file and read the relevant lines 
    f = open(fname) 
    head = f.readlines()[:8] 
    f.close() 
    
    # Get important stuff 
    cols = head[0].split() 
    
    # Put others lines in comments 
    comments = head[1:8] 
    
    return (cols, comments) 

def checkValue(value): 
    # Check if value should be a float  
    # otherwise I input a 0
    # It could also be flagged as missing  -> masked array
    if value == "---": 
        value = 0 #if I had wanted to mask the missing values, I would use MA.masked \
    else: 
        value = float(value) 
    return value 

def checkValid(items): 
    # Check if value should be a float  
    # otherwise I input a 0
    # It could also be flagged as missing  -> masked array
    if items[5] == '1': 
        valid = True 
        if items[6] == '0' or items[7] == '0':
            print('Item is valid but class is not')
    else: 
        valid = False
        #print('Item invalid')
    return valid 

def readData(fname): 
    # Open file and read column names and data block 
    f = open(fname) 
    # strip is for ignoring white space 
    temp = f.readline().split(",")
    col_names = [item.strip() for item in temp]

    # Ignore commnets 
    for i in range(7):  
        f.readline()  
    data_block = f.readlines() 
    f.close() 
    # Create a data dictionary, containing 
    # a list of values for each variable 
    temp = {}
    data = {} 
    
    nvalid = 0;
    for line in data_block: 
        items = line.split(",") 
        
        if checkValid(items):
            nvalid += 1
    
    # Add an entry to the data dictionary for each column 
    for col_name in col_names: 
        data[col_name] = np.zeros(int(nvalid))
        
   
    # Loop through each value: append to each column 
    line_count = 0
    for line in data_block: 
        items = line.split(",") 
        
        if checkValid(items):
            
            for (col_count, col_name) in enumerate(col_names): 
                value = items[col_count] 
                #data[col_name][line_count] = float(value);
                data[col_name][line_count] = checkValue(value) 
            line_count += 1
          
            
            
    return data 


    