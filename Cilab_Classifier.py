# -*- coding: utf-8 -*-
"""
ファジィ識別器をsklern.base.Classificationで実装したファイル
最適化は出来ないので，インスタンス生成時にルール集合を渡すか，インスタンス生成後にファイルからセットしてください．
基本的に，MoFGBMLライブラリから出力された識別器をファイルで読み込み，reject optionを適用することを想定しています．

@author: kawano
"""

from CIlab_FuzzyFunction import FuzzyFunction
from CIlab_function import CIlab
from ThresholdOptimization import predict_proba_transformer
from Runner import runner
from sklearn.neighbors import KNeighborsClassifier
from ThresholdBaseRejection import SingleThreshold, ClassWiseThreshold, RuleWiseThreshold, SecondStageRejectOption
from ThresholdOptimization import ThresholdEstimator
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from matplotlib import pyplot as plt


class Rule():
    
    def __init__(self, antecedent, class_label = None, CF = None, winner = None):
        
        """
        コンストラクタ
        
        Parameters
        ----------
        antecedent : array
                     前件部のリスト
        
        class_label : integer
                      結論部クラス，シングルラベルのみ対応してます．
        
        CF : float
             ルール重み
        ----------
        """
        
        self.antecedent = antecedent
        
        self.class_label = class_label
        
        self.CF = CF
        
        self.winner = winner
        
        
    def culc_conseqent(self, X, y):
        
        """
        ルールの後件部の計算
        Ruleインスタント生成時にclass_labelまたはCFがNoneの場合は，この関数を使用してください．
        
        Parameters
        ----------
        X, y : ndarray
               Xは入力データ集合，yは教師ラベル集合
        ----------
        """
        sum_c = np.sum([FuzzyFunction.compatibility(self.antecedent, x) for x in X])
        
        if sum_c != 0:
            
            num_class = np.max(y) + 1
            
            c = [np.sum([FuzzyFunction.compatibility(self.antecedent, x) for x in X[y == h]]) / sum_c for h in range(num_class)]
            
            self.class_label = np.argmax(c)
            
            self.CF = 2 * c[self.class_label] - np.sum(c)
            
            return self
            
        self.class_label = -1
        
        self.CF = 0
        
        return self
        
        
    def to_String(self):
        
        return f"IF {self.antecedent} then Class is {self.class_label} with {self.CF}"
    
    def clear_winner(self):
        
        self.winner = 0
        return self


    
class FuzzyClassifier(ClassifierMixin):
    
    def __init__(self, ruleset = None):
        
        """
        コンストラクタ
        
        Parameters
        ----------
        ruleset : Ruleインスタンスのリスト
        ----------
        """
        
        self.ruleset = ruleset
            
            
    def set_ruleset_csv(self, fname):
        
        """
        ファジィ識別器をファイルから読み取り，ルールリストとして返す．
        前件部があれば，後件部を学習することで識別器を生成出来ます．
        その際は，各ルールに対して，culc_conseqent(X, y)を適用してください．
        
        Parameters
        ----------
        fname : string
                ルールリストの情報が書き込まれたファイル名
        ----------
        """
        rule_list = pd.read_csv(fname, header = None)
        
        antecedent_list = rule_list.iloc[:, 0:-3].to_numpy()
        
        CF_list = rule_list.iloc[:, -2].to_numpy()
        
        class_list = rule_list.iloc[:, -3].to_numpy()
        
        
        ruleset = [Rule(antecedent, class_label, CF) for antecedent, class_label, CF in zip(antecedent_list, class_list, CF_list)]
        
        self.ruleset = ruleset
        
    def get_ruleNum(self, winner = None):
        
        if winner is True:
            return len([rule for rule in self.ruleset if rule.winner > 0])
        
        return len(self.ruleset)
    
    def get_mean_rule_length(self):
        
        sum_ = sum([np.count_nonzero(np.array(rule.antecedent) != 0) for rule in self.ruleset])
        
        return sum_ / self.get_ruleNum()
            
    def get_winner_weighted_rule_length(self):
        
        winner_sum = sum([rule.winner for rule in self.ruleset])
        
        sum_ = sum([np.count_nonzero(np.array(rule.antecedent) != 0) * rule.winner for rule in self.ruleset])
        
        return sum_ / winner_sum
    
    def get_fuzzyset_rule_length(self):
        
        def culc_one_rule(rule):
            return sum([FuzzyFunction.get_fuzzyset_area(a) for a in rule.antecedent])
        
        return sum([culc_one_rule(rule) for rule in self.ruleset]) / self.get_ruleNum()
    
    def clear_winner(self):
        
        self.ruleset = [rule.clear_winner() for rule in self.ruleset]
        
        return self
    
    def predict(self, X, winner_count = True):
        
        """
        単一勝利ルール戦略で選択されたルールの結論部クラスを返す
        異なる結論部クラスを持つルールが等しく最大値を持つ場合は，Noneを返す
        
        return : ndarray
                 推論結果のリスト        
        """
        
        def _single_winner_strategy(x):
            
            compatibility_list = [FuzzyFunction.compatibility(rule.antecedent, x) for rule in self.ruleset]
            
            value_dict = [{"value" : compatibility * rule.CF,
                           "class_label" : rule.class_label} \
                          for compatibility, rule in zip(compatibility_list, self.ruleset)]
 
                
            max_value_index = [i for i, x in enumerate(value_dict) \
                               if x["value"] == max([value["value"] for value in value_dict])]
            
            max_value_dict = [value_dict[i] for i in max_value_index]

    
            if not all(value["class_label"] == max_value_dict[0]["class_label"] for value in max_value_dict):
                
                return None
            
            if winner_count == True:
                
                for index in max_value_index:
                    
                    self.ruleset[index].winner = self.ruleset[index].winner + 1
            
            return max_value_dict[0]["class_label"]
        
        X = np.array(X)
        
        self.clear_winner()
        
        return np.array([_single_winner_strategy(x) for x in X])
    
    
    def fit(self, X, y):
        
        """
        pythonで識別器を学習したい場合は実装してください．
        """
        
        return self
    

    def predict_proba(self, X, base = None):
        
        """
        各クラスに属する確率を返す．
        ファジィ識別器は単一勝利ルール戦略のため，勝利ルールの結論部クラスに属する確率のみ計算し，
        その他のクラスに属する確率は0として実装．
        
        baseでクラス毎またはルール毎の確信度を指定する．
        クラス毎の確信度がデフォルトで，ルール毎の場合は base = "rule"で指定してください．
        ルール毎の場合，確信度のリスト長がルール数となり，勝者ルールのインデックスに確信度が代入されます．
        
        また，確信度の計算は以下の論文を参照．
        勝者ルールと勝者ルールの結論部クラスとは異なる結論部クラスを持つルールの内，適合度とルール重みの積が
        最大のルールとの差分（勝者ルールの適合度とルール重みの積で正規化）を計算する．
        
        Reference
        --------------------------------------------------------------------------------------------------------
        Y. Nojima and H. Ishibuchi, “Multiobjective fuzzy genetics-based machine learning with a reject option,
        ” In Proc. of 2016 IEEE International Conference on Fuzzy Systems, pp. 1405-1412, Jul. 2016.
        --------------------------------------------------------------------------------------------------------
        
        return : ndarray
                 確信度のリスト（ただし，最終的な出力以外のクラスに対する確率は0埋めしています．）
        """
        
        def _predict_proba_pattern(x):
            
             rule_dict_list = [{"compatibility" : FuzzyFunction.compatibility(rule.antecedent, x),
                                "rule" : rule,
                                "id" : i} \
                               for i, rule in enumerate(self.ruleset)]
            
             for rule_dict in rule_dict_list:
                 
                 rule_dict["value"] = rule_dict["compatibility"] * rule_dict["rule"].CF
                
            
             winner_dict = rule_dict_list[np.argmax([rule_dict["value"] for rule_dict in rule_dict_list])]
            

            
             second_winner_candidate = list(filter(lambda x : x["rule"].class_label != winner_dict["rule"].class_label, rule_dict_list))
            
             second_winner_dict = second_winner_candidate[np.argmax([rule_dict["value"] for rule_dict in second_winner_candidate])]
             
             
             if base == "rule":
                 
                 proba = np.zeros(max([rule_dict["id"] for rule_dict in rule_dict_list]) + 1)
                 
                 if winner_dict["value"] == 0:
                 
                     return proba
             
                 proba[winner_dict["id"]] = (winner_dict["value"] - second_winner_dict["value"]) / winner_dict["value"]
             
                 return proba
        
        
             proba = np.zeros(max([rule_dict["rule"].class_label for rule_dict in rule_dict_list]) + 1)
        
             if winner_dict["value"] == 0:
                 
                 return proba
 
             proba[winner_dict["rule"].class_label] = (winner_dict["value"] - second_winner_dict["value"]) / winner_dict["value"]
        
             return proba
    
        
        X = np.array(X)
         
        return np.array([_predict_proba_pattern(x) for x in X])
     
        
    def score(self, X, y):

        return len(X[self.predict(X) == y]) / len(X)
    
    
    def transform(self, X):
        
        return np.array(X)
    
    
    def to_dataFrame(self):
        
        columns = [f"antecedent{i}" for i in range(len(self.ruleset[0].antecedent))]
        
        columns.append("CF")
        
        columns.append("class_label")
        
        df = pd.DataFrame(data = [rule.antecedent + [rule.CF, rule.class_label] for rule in self.ruleset],
                          columns = columns)
    
        return df
    
    def winner_count(self, X, isReject = None):
        
        self.clear_winner()
        
        if isReject is None:
            
            isReject = np.ones(len(X), dtype=np.bool_) * False
            
        X_accept = X[~isReject]
        
        self.predict(X_accept, winner_count = True)
        
        return self
    
        
    def plot_rule(self, fname = None):
        
        df = self.to_dataFrame()
        
        X = np.arange(0, 1.01, 0.01)
        
        def colors_setting(class_label):
            
            if class_label == 0:
                return "tab:blue"
            
            if class_label == 1:
                return "tab:orange"
            
            if class_label == 2:
                return "tab:green"
        
        nrows = df["class_label"].max() + 1
        

        for i in range(len(self.ruleset[0].antecedent)):
            
            # plt.subplots_adjust(wspace=1, hspace=0.3)
            fig, axes = plt.subplots(nrows=nrows, ncols=1, squeeze=False, tight_layout=True)
            # fig.suptitle(f'attribute{i}', fontsize = 10)

            for ruleID, Fuzzyset in enumerate(df[f"antecedent{i}"]):
            
                class_label = self.ruleset[ruleID].class_label 
                
                Y = FuzzyFunction.memberships(Fuzzyset, X)
                
                axes[class_label, 0].plot(X, Y, c = colors_setting(class_label))
                axes[class_label, 0].set_xlim(0, 1)
                axes[class_label, 0].set_ylim(-0.02, 1.02)
                axes[class_label, 0].set_box_aspect(0.5)
                axes[class_label, 0].get_xaxis().set_visible(False)
                axes[class_label, 0].get_yaxis().set_visible(False)
            
            if fname != None:
                plt.savefig(f"{fname}_attribute{i}.png", dpi = 300)
 
            else:
                plt.show()
        return
        
    
    
def program_experiment_3():
    
    fname_train = "..\\dataset\\cilabo\\kadai5_pattern1.txt"
    
    fname_test = "..\\dataset\\cilabo\\kadai5_pattern1.txt"
    
    X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, "numpy")
    
    
    fuzzyset_ID = [i for i in range(28)]
    
    ruleset = [Rule([i, j]).culc_conseqent(X_train, y_train) for i in fuzzyset_ID for j in fuzzyset_ID]
    
    ruleset = list(filter(lambda x : x.CF > 0, ruleset))
    
    fuzzyClassifier = FuzzyClassifier(ruleset)
    
    print(fuzzyClassifier.get_mean_rule_length())
    
    
# program_experiment_3()
    
def experiment_3():
    
    fname_train = "..\\dataset\\cilabo\\kadai5_pattern1.txt"
    
    fname_test = "..\\dataset\\cilabo\\kadai5_pattern1.txt"
    
    X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, "numpy")
    
    
    
    antecedent_list = [[0, 5],
                        [3, 4],
                        [4, 3],
                        [4, 4],
                        [5, 0]]
    
    ruleset = [Rule(antecedent).culc_conseqent(X_train, y_train) for antecedent in antecedent_list]
    
    fuzzyClassifier = FuzzyClassifier(ruleset)
    
    fuzzyClassifier = fuzzyClassifier.winner_count(X_train)
    
    print(fuzzyClassifier.score(X_train, y_train))
    
    
    
    
if  __name__ == "__main__":
    
    experiment_3()
    
    # dataset = "pima"
    
    # fname_train = f"../dataset/{dataset}/a0_0_{dataset}-10tra.dat"
                 
    
    # fname_test = f"../dataset/{dataset}/a0_0_{dataset}-10tst.dat"
    
    
    # X_train, X_test, y_train, y_test = CIlab.load_train_test(fname_train, fname_test, "numpy")
    
    # run = runner(dataset, "RO-test", "trial00-v2", fname_train, fname_test)
    
    # rr = 0
    # cc = 0
    
    # fuzzy_cl = f"../results/MoFGBML_Basic/{dataset}/trial{rr}{cc}/VAR-0000600000.csv"
    
    
    # fuzzyClassifier = FuzzyClassifier()
    
    
    # best_model = fuzzyClassifier
    
    # param = {"kmax" : [200], "Rmax" : [0], "deltaT" : [0.001]}
    
    # # pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(best_model)),
    # #                           ('estimator', ClassWiseThreshold())])
    
    # pipe = Pipeline(steps = [('predict_proba_transform', predict_proba_transformer(best_model, base = "rule")),
    #                           ('estimator', RuleWiseThreshold(best_model.ruleset))])
    
    
    # second_model = KNeighborsClassifier()
    
    # thresh_estimator = ThresholdEstimator(pipe, param)
    
    # second_RO = SecondStageRejectOption(thresh_estimator, second_model)
    
    # run.run_second_stage(pipe, ParameterGrid(param), second_model, "train-rule.csv", "test-rule.csv")
