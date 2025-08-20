# -*- coding: utf-8 -*-

"""
@author: kawano
"""
import numpy as np
import pandas as pd
import os



class ThresholdBaseRejectOption():
    
    """
    抽象クラス
    sklearnの推論器クラス
    """
    
    def isReject(self, predict_proba_, threshold):
       
        """
        予測の確信度と閾値を比較し，棄却するかを返す．

        Parameters
        ----------
        predict_proba_ : ndarray
                         入力に対する予測の確信度
        threshold : ndarray
                    閾値，単一の閾値の場合でもlistにしてください．

        Returns
        -------
        ndarray
        入力を棄却するかを表すarray
        棄却する場合 : True, 棄却しない場合 : False
        """

        pass
    
    
    def zeros_threshold(self):
        
        """
        閾値初期化用の関数
        
        Returns
        -------
        ndarray
        閾値の長さ分の0埋めされたリスト

        """

        pass
        
        
    def accuracy(self, y, predict_, isReject):
        
        """
        accuracyを求める．
        
        Parameters
        ----------
        y : ndarray
            教師ラベル.

        predict_ : ndarray
                   予測ラベル
                   
        isReject : ndarray
                   predict_を棄却するかを表すarray
                   isReject()の返り値を渡すことを想定しています.

        Returns
        -------
        float : accuracy
        全ての予測ラベルを棄却する場合，accuracy は 1.0（誤識別率 0）として計算します．

        """
        len_accept = np.count_nonzero(~isReject)
        
        # if all patterns are rejected, accuracy is 1.0
        if len_accept == 0:
            
            return 1.0
        
        
        return np.count_nonzero((predict_ == y) & (~isReject)) / len_accept 
    
    
    def rejectrate(self, isReject):
        
        """
        reject rateを求める

        Parameters
        ----------
        isReject : ndarray
                   predict_を棄却するかを表すarray
                   isReject()の返り値を渡すことを想定しています.

        Returns
        -------
        float
        reject rateを返す．

        """
        
        return np.count_nonzero(isReject) / len(isReject)


    def fit(self, predict_proba_, y):
        
        return self
        
    
    def predict(self, predict_proba_, reject_option = True):
        
        """
        入力に対して，棄却する場合はNone，確信度が高い場合は予測ラベルを返す関数
        現状バグってます．

        Parameters
        ----------
        predict_proba_ : ndarray
                        入力に対する予測の確信度

        Returns
        -------
        list
        予測ラベル or None

        """
        if reject_option:
        
            return [np.argmax(proba) if ~self.isReject(proba, self.threshold) else None for proba in predict_proba_]
        
        

        return [np.argmax(proba) if not all(proba == 0) else None for proba in predict_proba_]


    def transform(self, X):
        
        return X
    

        
        
class SingleThreshold(ThresholdBaseRejectOption):
    
    """
    単一の閾値に基づく棄却オプション
    全てのパターンを１つの閾値で棄却の判定を行う．
    
    Reference
    --------------------------------------------------------------------------------
    C. K. Chow, “On optimum error and reject tradeoff,
    ” IEEE Trans. on Inform. Theory, vol. 16, pp.41-46, Jan. 1970.
    --------------------------------------------------------------------------------
    """
    
    def __init__(self):
        
        pass
    

    def zeros_threshold(self, y = None):
        
        return np.zeros(1)


    def isReject(self, predict_proba_, threshold):
        
        return np.max(predict_proba_, axis = 1) < threshold
        
    

class ClassWiseThreshold(ThresholdBaseRejectOption):
    
    """
    クラス毎の閾値に基づく棄却オプション
    
    Reference
    --------------------------------------------------------------------------------
    G. Fumera, F. Roli, and G. Giacinto, “Reject option with multiple thresholds,
    ” Pattern Recognition, vol. 33, no. 12, pp. 2099-2101, Dec. 2000.
    --------------------------------------------------------------------------------
    """
    
    def __init__(self):
        
        pass
                   
        
    def zeros_threshold(self, y):
        
        return np.zeros(max(y) + 1)
        
    
    def isReject(self, predict_proba_, threshold):
        
        index_list = np.argmax(predict_proba_, axis = 1)
        
        return np.array([proba[index] < threshold[index] for proba, index in zip(predict_proba_, index_list)])
    
    
class RuleWiseThreshold(ThresholdBaseRejectOption):
    
    """
    ルール毎の閾値に基づく棄却オプション
    現状では，CIlab_Classifierでしか使用できません．
    その他のルールベースの識別器で使用する際には，ルール毎の確信度を返す関数を実装してください．
    
    コンストラクタに使用するルールリストを渡してください．
    
    このクラスで使用するpredict_proba関数はクラス毎の確信度ではなく，ルール毎の確信度を返します．
    
    Reference
    --------------------------------------------------------------------------------
    川野弘陽，Eric Vernon，増山直輝，能島裕介，石渕久生，
    「複数の閾値を用いた棄却オプションの導入におけるファジィ識別器への影響調査」，
    インテリジェント・システム・シンポジウム 2021，オンライン，9 月，2021.
    --------------------------------------------------------------------------------
    """
    
    def __init__(self, ruleset):
        
        self.ruleset = np.array(ruleset)
                   
        
    def zeros_threshold(self, y = None):
        
        return np.zeros(len(self.ruleset))
    
    
    def isReject(self, predict_proba_, threshold):
            
        # proba_idx_list = np.argmax(predict_proba_, axis = 1)
        # old = np.array([proba[proba_idx] < threshold[proba_idx] for proba, proba_idx in zip(predict_proba_, proba_idx_list)]).flatten()
 
        # if not all(new == old):
        #     print(threshold)
        
        index_list = np.argmax(predict_proba_, axis = 1)
        
        return np.array([proba[index] < threshold[index] for proba, index in zip(predict_proba_, index_list)])
        
    

    def predict(self, predict_proba_, reject_option = True):
        
        winner_rule_id = np.argmax(predict_proba_, axis = 1)
    
        if reject_option:
        
            return [self.ruleset[np.argmax(proba)] \
                    if ~self.isReject(proba, self.threshold) else None for proba in predict_proba_]
                
        return np.array([self.ruleset[np.argmax(proba)].class_label if not all(proba == 0) else None for proba in predict_proba_])


class SecondStageRejectOption():
    
    """  
    ThresholdBaseRejectoptionでは，パターンの確信度と閾値から，棄却の判定を行うが，
    本クラスは，パターンと閾値から棄却の判定を行う．
    
    ２段階棄却オプションでは，探索した閾値に基づいて棄却と判定されたパターンに対して
    ファジィ識別器以外のモデルがファジィ識別器と同じ識別結果を出力した場合，棄却しない手法である．
    
    Reference
    -------------------------------------------------------------------------------------
    川野弘陽，Eric Vernon，増山直輝，能島裕介，石渕久生，
    「２段階棄却オプションを導入したファジィ識別器の精度と識別拒否のトレードオフ解析」，
    ファジィ・システム・シンポジウム 2022，オンライン，9 月，2022.
    -------------------------------------------------------------------------------------
    """
    
    def __init__(self, thresh_estimator, second_classifier):
        
        self.thresh_estimator = thresh_estimator
        
        self.second_classifier = second_classifier
        
        
    def isReject(self, X, threshold):
        
        predict_proba_ = self.thresh_estimator.pipe.transform(X)
        
        predict_ = self.thresh_estimator.pipe[-1].predict(predict_proba_, reject_option = False)
        
        return self.thresh_estimator.pipe[-1].isReject(predict_proba_, threshold) & (self.second_classifier.predict(X) != predict_)
    
    
    def accuracy(self, X, y, threshold = None):
        
        if threshold == None:
            
            threshold = self.thresh_estimator.threshold
            
        isReject = self.isReject(X, threshold)
        
        # if all patterns are rejected, accuracy is 1.0
        len_accept = np.count_nonzero(~isReject)
        
        if len_accept == 0:
            
            return 1.0
        
        predict_ = self.thresh_estimator.pipe[0].model.predict(X[~isReject])

        return np.count_nonzero(predict_ == y[~isReject]) / len_accept
    
    
    def rejectrate(self, X, threshold = None):
        
        if threshold == None:
            
            threshold = self.thresh_estimator.threshold
        
        return np.count_nonzero(self.isReject(X, threshold)) / len(X)
