# -*- coding: utf-8 -*-
"""
CIlab style datasetを使用する際の関数を集めたユーティリティクラス

@author: kawano
"""
import pandas as pd
import os
import numpy as np

class CIlab():
    
    # utility class of CI lab
    
    @staticmethod
    def load_cilab_style_dataset(fname):
       
        df = pd.read_csv(fname,
                         header = None,
                         skiprows = 1)
        
        df = df.dropna(how = 'all', axis = 1)
        
        columns = [f"attribute{i}" for i in range(len(df.columns) - 1)]
        
        columns.append("target")
        
        df.columns = columns
        
        return df

    
    @staticmethod
    def load_dataset(fname, type_ = "pandas"):
        
        df = CIlab.load_cilab_style_dataset(fname)
    
        X = df[[f"attribute{i}" for i in range(len(df.columns) - 1)]]
        
        y = df["target"]
        
        if type_ == "numpy":
            
            return X.to_numpy(), y.to_numpy()
        
        return X, y 
    
    
    @staticmethod
    def load_train_test(fname_train, fname_test, type_ = "pandas"):
        
        X_train, y_train = CIlab.load_dataset(fname_train, type_)
        
        X_test, y_test = CIlab.load_dataset(fname_test, type_)
        
        return X_train, X_test, y_train, y_test 
    
    
    @staticmethod
    def dataset_file(pass_, dataset, RR, CC):
        
        return f"{pass_}{dataset}\\a{RR}_{CC}_{dataset}-10tra.dat",\
               f"{pass_}{dataset}\\a{RR}_{CC}_{dataset}-10tst.dat" 
    
    
    @staticmethod
    def output_dict(dict_, output_dir, fname):
        
        if not os.path.exists(output_dir):
        
            os.makedirs(output_dir)
    
        f = open(output_dir + fname, 'w')
        
        for key,value in sorted(dict_.items()):
        	
            f.write(f"{key} : {value}\n")
        
        f.close()
        
        return
    
    def output_cilab_style_dataset(X, y, output_dir, fname):
        
        header = f"{len(X)}, {len(X[0])}, {max(y) + 1}"
        
        if not os.path.exists(output_dir):
        
            os.makedirs(output_dir)
            
        dataset = np.array([np.append(x, y) for x, y in zip(X, y)])
        
        np.savetxt(f"{output_dir}/{fname}",
                   dataset,
                   delimiter = ',',
                   newline = ',\n',
                   fmt = '%f',
                   header = header,
                   comments = "")
        
        return
    

    
