# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 14:50:45 2022

@author: kawano
"""
import os
import pandas as pd


def to_csv(result_list, output_dir, fname):
    
    
    if len(result_list[0]) == 2:
        columns = ["accuracy", "rejectrate"]
    
           
    if len(result_list[0]) == 3:
           
        columns = ["accuracy", "rejectrate", "threshold"]
        
    if len(result_list[0]) == 4:
        
        columns = ["accuracy", "rejectrate", "rule_num", "threshold"]
        
        
    df = pd.DataFrame(result_list, columns = columns)
    
    if not os.path.exists(output_dir):
        
        os.makedirs(output_dir)

    df.to_csv(output_dir + fname, index = False) 