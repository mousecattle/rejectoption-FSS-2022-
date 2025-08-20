# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 19:07:38 2022

@author: kawano
"""
import csv
from Cilab_Classifier import Rule, FuzzyClassifier
from Runner import runner
import numpy as np
import pandas as pd
from CIlab_function import CIlab

class FileInput():
    
    def __init__(self):
        
        pass
        
        
    def input_VAR(fname):
        
        rows = []
        
        with open(fname) as f:
            
            reader = csv.reader(f)
            
            group = []
            
            for row in reader:
                
                if len(row) != 0:
                    
                    group.append(row)
                    
                    continue
                
                rows.append(group)
                group = []
                
        rows = [ruleset[1:-1] for ruleset in rows if len(ruleset) != 0]
        
        pop = []
        
        start = 2
        
        add_info = 10
    
        num_attribute = len(rows[0][0]) - (start + add_info)
        
        for row in rows:
            
            ruleset = []
            
            for rule in row:
        
                ruleset.append(Rule(antecedent = [int(antecedent.strip()) for antecedent in rule[start : start + num_attribute]],
                                    CF = float(rule[start + num_attribute + 1]),
                                    class_label = int(rule[start + num_attribute + 3]),
                                    winner = int(rule[start + num_attribute + 9][52:-1])))
                
            pop.append(FuzzyClassifier(ruleset))
            
            
        return pop
    
    def input_train_VAR(fname, X_train, y_train):
        
        rows = []
        
        with open(fname) as f:
            
            reader = csv.reader(f)
            
            group = []
            
            for row in reader:
                
                if len(row) != 0:
                    
                    group.append(row)
                    
                    continue
                
                rows.append(group)
                group = []
                
        rows = [ruleset[1:-1] for ruleset in rows if len(ruleset) != 0]
        
        pop = []
        
        start = 2
        
        add_info = 10
    
        num_attribute = len(rows[0][0]) - (start + add_info)
        
        for row in rows:
            
            ruleset = []
            
            for rule in row:
        
                ruleset.append(Rule([int(antecedent.strip()) for antecedent in rule[start : start + num_attribute]]).culc_conseqent(X_train, y_train))
                
            pop.append(FuzzyClassifier(ruleset))
            
            
        return pop
    
    def input_classify(fname):
        
        rule_list = pd.read_csv(fname, header = None)
        
        antecedent_list = rule_list.iloc[:, 0:-3].to_numpy()
        
        CF_list = rule_list.iloc[:, -2].to_numpy()
        
        class_list = rule_list.iloc[:, -3].to_numpy()
        
        ruleset = [Rule(antecedent, class_label, CF) for antecedent, class_label, CF in zip(antecedent_list, class_list, CF_list)]
        
        return FuzzyClassifier(ruleset)
    
    def to_dataFrame(fuzzy_clf_list, X_train, X_test, y_train, y_test):
        
        columns = ["rule_num", "acc_train", "acc_test", "rule_length"]
        
        data = [[clf.get_ruleNum(),
                   clf.score(X_train, y_train),
                   clf.score(X_test, y_test),
                   clf.get_mean_rule_length()] for clf in fuzzy_clf_list]
        
        return pd.DataFrame(data = data, columns = columns)
    
    
    def best_classifier(fname, X_train, y_train):
        
        fuzzy_clf_list = FileInput.input_VAR(fname)

        return fuzzy_clf_list[np.argmax([fuzzy_clf.score(X_train, y_train) for fuzzy_clf in fuzzy_clf_list])]
         

    
    def is_all_class(fuzzy_clf, num_class):
        
        rule_class = set(np.array([rule.class_label for rule in fuzzy_clf.ruleset]))
        
        full_set = set(np.arange(num_class))
        
        return len(full_set.difference(rule_class)) == 0
    
    

if __name__ == "__main__":
    
    dataset = "pima"
    
    # fname_train = f"../dataset/{dataset}/a1_4_{dataset}-10tra.dat"
    
    # fname_test = f"../dataset/{dataset}/a1_4_{dataset}-10tst.dat"
    
    # X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, "numpy")
    
    rr = 1
    cc = 4
    
    algorithm = "MoFGBML_Basic_28set"
    
    file = f"../results/{algorithm}/{dataset}/trial{rr}{cc}/VAR-0000600000.csv"
    
    clf_list = FileInput.input_VAR(file)
    clf = list(filter(lambda x: x.get_ruleNum() == 8, clf_list))[0]
    clf.plot_rule(fname = f"../results/plots/pop/{dataset}/{dataset}_{algorithm}")
    
    print(clf.get_mean_rule_length())
    
    algorithm = "MoFGBML_Basic"
    
    file = f"../results/{algorithm}/{dataset}/trial{rr}{cc}/VAR-0000600000.csv"
    
    clf_list = FileInput.input_VAR(file)
    clf = list(filter(lambda x: x.get_ruleNum() == 8, clf_list))[0]
    
    print(clf.get_mean_rule_length())
        
