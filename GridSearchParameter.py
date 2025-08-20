# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 16:36:27 2023

@author: kawano
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import  GridSearchCV
import pandas as pd
import os

class GridSearch():
    
    def __init__():
        
        pass
    
    def get_parameter(algorithm):
        
        random_state = 2022
    
        if algorithm == "kNN":
            
            model = KNeighborsClassifier()
        
            param = {"n_neighbors" : [2, 3, 4, 5, 6]}
        
            return model, param
        
        if algorithm == "DecisionTree":
            
            model = DecisionTreeClassifier(random_state = random_state)
             
            param = {"max_depth" : [5, 10, 20]}
        
            return model, param
        
        if algorithm == "Adaboost":
            
            model = AdaBoostClassifier(algorithm = 'SAMME', random_state = random_state)
            
            param = {}
        
            return model, param
        
        if algorithm == "NaiveBayes":
            
            model = GaussianNB()
            
            param = {}
        
            return model, param
        
        if algorithm == "GaussianProcess":
            
            model = GaussianProcessClassifier(random_state = random_state)
            
            param = {}
        
            return model, param
        
        if algorithm == "MLP":
            
            model = MLPClassifier(random_state = random_state, max_iter = 10000)
            
            param = {"activation" : ["relu"], "alpha" : [10 ** i for i in range(-5, 3)]}
        
            return model, param
        
        if algorithm == "RF":
            
            model = RandomForestClassifier(random_state = random_state)
            
            param = {"max_depth" : [5, 10, 20]}
        
            return model, param
        
        if algorithm == "LinearSVC":
            
            model = LinearSVC(random_state = random_state)
            
            param = {"C" : [2 ** i for i in range(-5, 16)]}
        
            return model, param
        
        if algorithm == "RBFSVC":
            
            model = SVC(random_state = random_state)
            
            param = {"C" : [2 ** i for i in range(-10, 11)]}
        
            return model, param
            
        
        
    def run_grid_search(algorithm, X, y, output_dir, fname, cv = 10):
        
        model, param = GridSearch.get_parameter(algorithm)
    
        if  param != None:
        
            gscv = GridSearchCV(model, param, cv = cv, verbose = 0, n_jobs = 3)
        
            gscv.fit(X, y)
            
            gs_result = pd.DataFrame.from_dict(gscv.cv_results_)
            
            if not os.path.exists(output_dir):
        
                os.makedirs(output_dir)
        
            gs_result.to_csv(output_dir + fname)
            
            return gscv.best_estimator_
        
        return model
    
    # def load_parameter_gs_results(fname):
        
    #     df = pd.read_csv(fname)
        
    #     return
    
    
if __name__ == "__main__":
    
    
    
    # algorithm = "GaussianProcess"
    
    # dataset = "pima"
    
    # fname_train = f"..\\dataset\\{dataset}\\a0_0_{dataset}-10tra.dat"
                 
    # fname_test = f"..\\dataset\\{dataset}\\a0_0_{dataset}-10tst.dat"
   
    # X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, "numpy")
   
    # model = GridSearch.run_grid_search(algorithm, X_train, y_train)
    pass

