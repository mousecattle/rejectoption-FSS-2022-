# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 19:06:50 2022

@author: kawano
"""
from joblib import Parallel, delayed
from ThresholdBaseRejection import SecondStageRejectOption
from CIlab_function import CIlab
from ThresholdOptimization import ThresholdEstimator
import file_output as output
from GridSearchParameter import GridSearch
import copy

class runner():
    

    
    def __init__(self,
                 dataset,
                 algorithmID,
                 experimentID,
                 fname_train,
                 fname_test):
        
        """
        コンストラクタ
        
        Parameter
        ---------
        MoFGBMLライブラリの仕様に合わせています．
        dataset : dataset name : string, ex. "iris"
        
        algorithmID : string
                      "result"直下のディレクトリ名 
        
        experimentID : string
                       出力ファイルのデイレクトリ名, 
                       出力ファイルは "result\\algorithmID\\experimentID に出力されます．
        
        file_train : string 
                     学習用データのファイル名
        
        file_test : string
                    評価用データのファイル名
        """
    
        self.dataset = dataset
        self.algorithmID = algorithmID
        self.experimentID = experimentID
        self.X_train, self.X_test, self.y_train, self.y_test = CIlab.load_train_test(fname_train, fname_test, type_ = "numpy")
        self.output_dir = f"../results/threshold_base/{self.algorithmID}/{self.dataset}/{self.experimentID}/"
    

    def run_second_stage(self, pipe, params, second_models, train_file, test_file, core = 5):
        
        """
        run function
        2段階棄却オプションのARCs(Accuracy-Rejection Curves)で必要なデータを出力する関数.
        ベース識別器がファジィ識別機以外だとエラーを起こすので注意！
        他のモデルを使用する際は，ルール数の計算をしないようにしてください．
        
        Parameter
        ---------
        pipe : Pipeline module
               ステップ：predict_proba_transfomer，ThresholdBaseRejectOption
        
        params : パラメータ辞書のリスト, 
                 辞書のキーは，"kmax", "Rmax", "deltaT"にしてください．
        
        second_model : sklearn.ClassifierMixin
                       sklearnの識別器で使用される関数を実装したモデル
                       2段階目の判定で用いるモデル．
        
        train_file : file name of result for trainning data, result is accuracy, reject rate, threshold
        
        test_file : file name of result for test data
        """
        
        def _run_one_search_threshold(param):
            
            return ThresholdEstimator(copy.deepcopy(pipe), param).fit(self.X_train, self.y_train)
 
        
        # 閾値の探索
        
        result_list = Parallel(n_jobs = core)(delayed(_run_one_search_threshold)(param) for param in params)
        # result_list = [_run_one_search_threshold(param) for param in params]

        train_score_result = [result.score(self.X_train, self.y_train) for result in result_list]
        # 学習用データの結果をまとめて出力
        train_result = [[score["accuracy"],
                         score["rejectrate"],
                         result.pipe[0].model.get_ruleNum(winner = True),
                         result.threshold]\
                         for result, score in zip(result_list, train_score_result)]
        
        output.to_csv(train_result, self.output_dir, train_file)
        
        # 評価用データの結果をまとめて出力
        test_score_result = [result.score(self.X_test, self.y_test) for result in result_list]
           
        test_result = [[score["accuracy"],
                       score["rejectrate"],
                       result.pipe[0].model.get_ruleNum(winner = True),
                       result.threshold]\
                       for result, score in zip(result_list, test_score_result)]
        
        output.to_csv(test_result, self.output_dir, test_file)
        
        
        # 2段階棄却オプション，やってることは上と同じ
        for second_model in second_models:
            
            model = GridSearch.run_grid_search(second_model,self.X_train, self.y_train, f"{self.output_dir}/{second_model}/", f"gs_result_{second_model}.csv")

            model.fit(self.X_train, self.y_train)
            
            second_RO_list = [SecondStageRejectOption(thresh_estimator, model) for thresh_estimator in result_list]
            
            second_RO_train_result = [[second_RO.accuracy(self.X_train, self.y_train),
                                       second_RO.rejectrate(self.X_train),
                                       second_RO.thresh_estimator.pipe[0].model.get_ruleNum(winner = True),
                                       second_RO.thresh_estimator.threshold]\
                                       for second_RO in second_RO_list]
            
            output.to_csv(second_RO_train_result, f"{self.output_dir}/{second_model}/", "second-" + train_file)
            
            second_RO_test_result = [[second_RO.accuracy(self.X_test, self.y_test),
                                      second_RO.rejectrate(self.X_test),
                                      second_RO.thresh_estimator.pipe[0].model.get_ruleNum(winner = True),
                                      second_RO.thresh_estimator.threshold] \
                                      for second_RO in second_RO_list]
            
            output.to_csv(second_RO_test_result, f"{self.output_dir}/{second_model}/", "second-" + test_file)
            
        
    def output_const(self, dict_):
        
        CIlab.output_dict(dict_, self.output_dir, "Const.txt")
    
    
def main():
    
    return
    # dataset = "pima"
    
    # param = {"max_depth" : [5, 10, 20]}
    
    # model = DecisionTreeClassifier()
    
    # run = runner(dataset,
    #              "RO-test",
    #              "trial00-v2",
    #              f"..\\dataset\\{dataset}\\a0_0_{dataset}-10tra.dat",
    #              f"..\\dataset\\{dataset}\\a0_0_{dataset}-10tst.dat")
   
    
    # best_model = run.grid_search(model, param)


    # param = {"kmax" : [700], "Rmax" : [0.5], "deltaT" : [0.001]}
    
    # pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(best_model)),
    #                           ('estimator', ClassWiseThreshold())])
    
    
    # second_model = KNeighborsClassifier()
    
    # run.run_second_stage(pipe, ParameterGrid(param), second_model, "train_gomi.csv", "test_gomi.csv")
    
    
if __name__ == "__main__":
    
    main()
