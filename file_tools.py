#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:08:38 2018

@author: zeynep
"""

import numpy.ma as MA 
import numpy as np

def find_between( s, first, last ):
    """
    Find the part of string s, which is between first and last
    firstfirstHERElast returns firstHERE
    If you want to go deeper use the other function below
    """
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
    
def find_between_r( s, first, last ):
    """
    This one goes deeper. 
    For example
    firstfirstHERElast returns HERE
    """
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""
    
def readHeader(fname): 
    # Open the file and read the relevant lines 
    f = open(fname) 
    head = f.readlines()[:8] 
    f.close() 
    
    # Get important stuff 
    fields = (head[0].replace(',', '')).split() 
    
    ## Put others lines in comments, if necessary
    # comments = head[1:8] 
    
    return fields 

def checkValue(value): 
    # Check if value should be a float  
    # otherwise I input a 0
    # It could also be flagged as missing  -> masked array
    if value == "---": 
        value = 0 #if I had wanted to mask the missing values, I would use MA.masked \
    elif value.isdigit(): 
        value = float(value) 
    else:
        value = value.replace('\n', '')
        # this is a little annoying
        # after removing the line break, the new string (ie value) may be a digit
        # so i check it one last time and cast to float, if necessary
        if value.isdigit():
            value = float(value)
    
    return value 

def checkValid(items): 
    if items[7] == '1': # this one is hardcoded so be careful
        valid = True 
        if items[8] == '0' or items[9] == '0':
            print('Item is valid but class is not')# this should never happen
    else: 
        valid = False
        #print('Item invalid')
    return valid 

def readExpInfo(fname): 
    
    # Create a data dictionary, containing 
    # a list of values for each variable 
    exp_info = {} 
    fields = readHeader(fname)
    for field in fields: 
        exp_info[field] = []
  
    # Ignore comments 
    f = open(fname) 
    for i in range(8):  # 7 or 8? -> 8
        f.readline()  
    data_block = f.readlines() 
    f.close() 
   
    # Loop through each value: append to each column 
    line_count = 0
    for line in data_block: 
        items = line.split(",") 
        
        if checkValid(items):
            
            for (f, field) in enumerate(fields): 
                value = items[f] 
                exp_info[field].append( checkValue(value) )
            line_count += 1
                      
    return exp_info 


    