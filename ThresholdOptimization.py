import numpy as np
import copy
from sklearn.pipeline import Pipeline
import os
import pandas as pd
import itertools

class ThresholdEstimator():

    """
    this class is threshold estimator of fumera's method
    
    Reference
    --------------------------------------------------------------------------------------------------------------------
    G. Fumera and F. Roli, “Multiple reject thresholds for improving classification reliability,” 
    In Proceedings of the Joint IAPR International Workshopson Advances in Pattern Recognition, pp. 863–871, Aug. 2000.
    --------------------------------------------------------------------------------------------------------------------
    """
    
    def __init__(self,
                 pipe : Pipeline,
                 param):
        
        """
        コンストラクタ
        
        Parameter
        --------
        pipe  : Pipeline
                predict_proba_transformerの前処理＋ThresholdBaseRejectOptionの推論のパイプライン
        
        param : dictionary,
                key : "kmax", "Rmax", "deltaT"
        --------
        """
        
        self.param = param
        self.pipe = pipe
    
    
    def fit(self, X, y):
        
        """
        パイプラインに適用させた後，self.paramに従い閾値の探索を行う．
        """
        self.pipe.fit(X, y)
        
        self._run_search(X, y)
        
        return self
    
    
    
    def _run_search(self, X, y):
        
        def list_increment(list_, ite, value):
        
            list_copy = copy.deepcopy(list_)
            
            list_copy[ite] = list_copy[ite] + value
    
            return np.array(list_copy)
        
            
        def run_one_thresh(threshold):
            
            isReject = self.pipe[-1].isReject(self.predict_proba_, threshold)
            
            rejectrate = self.pipe[-1].rejectrate(isReject)
            
            accuracy = self.pipe[-1].accuracy(y, self.predict_, isReject)                         
               
            return {"threshold" : threshold, "accuracy" : accuracy, "rejectrate" : rejectrate, "isReject" : isReject}
            
            
        self.threshold = self.pipe[-1].zeros_threshold(y)
        
        self.predict_proba_ = self.pipe.transform(X)
      
        self.predict_ = self.pipe[-1].predict(self.predict_proba_, reject_option = False)
        
        isReject = self.pipe[-1].isReject(self.predict_proba_, self.threshold)
        
        self.accuracy = self.pipe[-1].accuracy(y, self.predict_, isReject)
                            
        self.rejectrate = 0.0
        
        self.isReject = np.ones(len(X), dtype=np.bool_) * False
        
        search_idx = np.arange(len(self.threshold))
        
        ignore_idx = np.array([])
        
        while(True):
            
            thresh_candidate = []
            
            for idx in np.setdiff1d(search_idx, ignore_idx):
                
                search_thresh = [list_increment(self.threshold, idx, k * self.param["deltaT"]) for k in range(self.param["kmax"] + 1)]
                
                search_thresh = list(filter(lambda x : all(x <= 1 + self.param["deltaT"]), search_thresh))
                
                result = [run_one_thresh(thresh) for thresh in search_thresh]
                
                thresh_candidate.append(result)
                
                if result[-1]["rejectrate"] == self.rejectrate:
                    
                    ignore_idx = np.append(ignore_idx, idx)
                    
                    
            thresh_candidate = list(itertools.chain.from_iterable(thresh_candidate))
                
            
            # 前回のaccuracyを改善しかつ，制約Rmaxを超えないような閾値の候補を求める．
            thresh_candidate = list(filter(lambda x : (x["accuracy"] > self.accuracy) and (x["rejectrate"] < self.param["Rmax"]), thresh_candidate))
            
            for candidate in thresh_candidate:
                
                candidate["value"] = (candidate["accuracy"] - self.accuracy) / (candidate["rejectrate"] - self.rejectrate)
            
            # 制約内でaccuracyを改善できる閾値がなければ終了
            if len(thresh_candidate) == 0:
                
                break
            
            thresh_candidate = thresh_candidate[np.argmax([x["value"] for x in thresh_candidate])]
      
            # 更新
            self.accuracy = thresh_candidate["accuracy"]
            self.threshold = copy.deepcopy(thresh_candidate["threshold"])
            self.rejectrate = thresh_candidate["rejectrate"]
            self.isReject = thresh_candidate["isReject"]
        
        return self
            
    
    def score(self, X, y, threshold = None):
        
        """
        探索した閾値に基づき，データセットのaccuracy，及びreject rateを求める．
        
        return : dict
                 key : "accuracy", "rejectrate"
        """        
        if threshold == None:
            
            threshold = self.threshold
            
                
        isReject = self.func_isReject(X)
        
        predict = self.pipe[0].model.predict(X[~isReject])
        
        len_accept = np.count_nonzero(~isReject)
        
        # if all patterns are rejected, accuracy is 1.0
        accuracy = 1.0
        
        if len_accept != 0:
            
            accuracy = np.count_nonzero(predict == y[~isReject]) / len_accept 
        
        return {"accuracy" : accuracy,
                "rejectrate" : self.pipe[-1].rejectrate(isReject)}
    
    
    def func_isReject(self, X):
        
        predict_proba_ = self.pipe.transform(X)
        
        return self.pipe[-1].isReject(predict_proba_, self.threshold)


class predict_proba_transformer():
    
    """
    Transfomerクラス（前処理を行うクラス）
    入力ベクトルを入力ベクトルに対するモデルの確信度に変換する．
    
    ここでは，sklearn.predict_proba()を使用する事を想定しています．
    """
    
    def __init__(self, _model, base = None):
        
        """
        コンストラクタ
        
        Parameter
        ---------
        model : sklearn.ClassifierMixin
                sklearnの識別器で使用される関数を実装したモデル
                
        base : string
               確信度のベースを指定する．
               デフォルトは各クラスに対する確信度に変換するが，base = "rule"と指定することで各ルールに対する確信度に変換する．
               "rule"を使用する場合は，現在はCIlab_Classifier.FuzzyClassifierをモデルとして使用して下さい．
        """
        
        self.model = _model
        
        self.base = base
    
    
    def fit(self, X, y):
        
        self.model.fit(X, y)
        
        return self
    
    
    def transform(self, X):
        
         if self.base != None:
             
             return self.model.predict_proba(X, base = self.base)
         
         predict_proba_ = self.model.predict_proba(X)
         
         # for proba in predict_proba_:
             
         #     proba[np.argmin(proba)] = 0
             
         return predict_proba_
     
        
  


    
    
        
    