# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:31:26 2022

@author: kawano
"""

from functools import reduce
from operator import mul
import numpy as np
from matplotlib import pyplot as plt

    
class FuzzyFunction():
    
    def __init__(self):
        pass
    
    
    def transform_fuzzyID(fuzzyset_ID):
        """
        三角型メンバシップの割当を行う
        0 : don'tc are, 1 : (2, 1) -> 2分割の1つ目のメンバシップ
        2~7分割まで対応しています．

        Parameters
        ----------
        fuzzyset_ID : int
                      メンバシップに割り当てられたID

        Returns
        -------
        int
            分割数
        int
            分割の中で頂点何番目に0に近いか.

        """
        
        if fuzzyset_ID == 0:
            
            return 0, 0
        
        if fuzzyset_ID <= 2:
    
            return 2, fuzzyset_ID % 3
        
        if fuzzyset_ID <= 5:
            
            return 3, fuzzyset_ID % 3 + 1
        
        if fuzzyset_ID <= 9:
            
            return 4, fuzzyset_ID % 5
        
        if fuzzyset_ID <= 14:
            
            return 5, fuzzyset_ID % 9
        
        if fuzzyset_ID <= 20:
            
            return 6, fuzzyset_ID % 14
        
        if fuzzyset_ID <= 27:
            
            return 7, fuzzyset_ID % 20
        
        
    def membership(fuzzyset_ID, x):
        """
        三角型メンバシップ関数の

        Parameters
        ----------
        fuzzyset_ID : int
                      ファジィ集合に割り当てられたID
            
        x : ndarray
            入力ベクトル.

        Returns
        -------
        float
            メンバシップ値.

        """
    
        K, k = FuzzyFunction.transform_fuzzyID(fuzzyset_ID)
        
        if(k == 0):
            
            return 1
        
        a = (k - 1) / (K - 1)
        
        b = 1 / (K - 1)
        
        return max(1 - abs(a - x) / b, 0)
    
    def memberships(fuzzySetID, X):
        
        return np.array([FuzzyFunction.membership(fuzzySetID, x) for x in X])
    

    def compatibility(antecedent, x):
        """
        適合度を求める

        Parameters
        ----------
        antecedent : list
                     前件部のリスト，各要素はファイジィ集合のIDを表す.
                     
        x : ndarray
            入力ベクトル

        Returns
        -------
        float
            適合度

        """
        
        membership_values = [FuzzyFunction.membership(fuzzyset, x_i) for fuzzyset, x_i in zip(antecedent, x)]
        
        return reduce(mul, membership_values)
    

    def plot_compatibility_heatmap(fuzzySetID1, fuzzySetID2, outputfile = None, isColorbar = False):
        
        x = np.arange(0, 1, 0.001)
        
        y = np.arange(0, 1, 0.001)
        
        X, Y = np.meshgrid(x, y)
        
        Z = [[FuzzyFunction.compatibility([fuzzySetID1, fuzzySetID2], [x_ij, y_ij]) for x_ij, y_ij in zip(x_i, y_i)] for x_i, y_i in zip(X, Y)]
        
        fig = plt.figure(figsize = (5, 5))
        
        plt.xlim(0, 1.02)
        
        plt.ylim(0, 1.02)
        
        plt.contourf(X, Y, Z, cmap = 'jet', alpha = 0.8)
        
        if isColorbar:
        
            plt.colorbar()
        
        if outputfile != None:
            
            plt.savefig(outputfile, dvi = 400)
            
    def get_fuzzyset_area(fuzzyset_ID):
        
        if fuzzyset_ID == 0:
            return 1.0
        
        if fuzzyset_ID <= 2:
            return 0.5
        
        if fuzzyset_ID <= 5:
            if fuzzyset_ID == 4:
                return 0.5
            return 0.25
        
        if fuzzyset_ID <= 9:
            if fuzzyset_ID == 7 or fuzzyset_ID == 8:
               return 1/3
            return 1/6
        
        if fuzzyset_ID <= 14:
            if fuzzyset_ID == 10 or fuzzyset_ID == 14:
                return 0.125
            return 0.25
        
        return "No assignment"
            
if __name__ == "__main__":
    
    FuzzyFunction.plot_compatibility_heatmap(0, 27)
