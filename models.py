#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import catboost
import xgboost
import lightgbm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# catboostによるモデル1 (Optunaにより最適化されたハイパーパラメーターを使用)
class model_cat1:
    def __init__(self):
        self.model = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        model = catboost.CatBoostRegressor(learning_rate = 0.03,
                                     max_depth = 10,
                                     n_estimators = 2000)
        model.fit(X_train, y_train)
        self.model = model
        
    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred

# catboostによるモデル2 
class model_cat2:
    def __init__(self):
        self.model = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        model = catboost.CatBoostRegressor(learning_rate = 0.03,
                                     max_depth = 7,
                                     n_estimators = 500)
        model.fit(X_train, y_train)
        self.model = model
        
    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred
    

# XGBoostによるモデル1 (Optunaにより最適化されたハイパーパラメーターを使用)
class model_xgb1:
    def __init__(self):
        self.model = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        model = xgboost.XGBRegressor(learning_rate = 0.03,
                                     max_depth = 7,
                                     n_estimators = 1000,
                                     alpha = 1.298962486458325,
                                     random_state = 1234)
        model.fit(X_train, y_train)
        self.model = model
        
    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred

# XGBoostによるモデル2
class model_xgb2:
    def __init__(self):
        self.model = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        model = xgboost.XGBRegressor(learning_rate = 0.05,
                                     max_depth = 20,
                                     n_estimators = 100,
                                     random_state = 1234)
        model.fit(X_train, y_train)
        self.model = model
        
    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred
    
# lightGBMによるモデル1 (Optunaにより最適化されたハイパーパラメータを使用)
class model_lgbm1:
    def __init__(self):
        self.model = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        params = {"task":"train",
                  "objective":"regression",
                  "learning_rate":0.03,
                  "max_depth":7,
                  "n_estimators":2000,
                  "alpha" : 9.687300617814392,
                  "metric":"root_mean_squared_error",
                  "random_state":1234}
        num_round = 10
        lgbm_train = lightgbm.Dataset(X_train, y_train)
        lgbm_valid = lightgbm.Dataset(X_val, y_val)
        self.model = lightgbm.train(params,
                                    num_boost_round=num_round,
                                    train_set = lgbm_train,
                                    valid_sets =lgbm_valid)
        
    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred

# lightGBMによるモデル2    
class model_lgbm2:
    def __init__(self):
        self.model = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        params = {"task":"train",
                  "objective":"regression",
                  "learning_rate":0.1,
                  "max_depth":5,
                  "n_estimators":200,
                  "alpha" : 9.687300617814392,
                  "metric":"root_mean_squared_error",
                  "random_state":1234}
        num_round = 10
        lgbm_train = lightgbm.Dataset(X_train, y_train)
        lgbm_valid = lightgbm.Dataset(X_val, y_val)
        self.model = lightgbm.train(params,
                                    num_boost_round=num_round,
                                    train_set = lgbm_train,
                                    valid_sets =lgbm_valid)
        
    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred

# lightGBMによるモデル3
class model_lgbm3:
    def __init__(self):
        self.model = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        params = {"task":"train",
                  "objective":"regression",
                  "learning_rate":0.05,
                  "max_depth":50,
                  "n_estimators":4000,
                  "alpha" : 9.687300617814392,
                  "metric":"root_mean_squared_error",
                  "random_state":1234}
        num_round = 10
        lgbm_train = lightgbm.Dataset(X_train, y_train)
        lgbm_valid = lightgbm.Dataset(X_val, y_val)
        self.model = lightgbm.train(params,
                                    num_boost_round=num_round,
                                    train_set = lgbm_train,
                                    valid_sets =lgbm_valid)
        
    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred  

# RandomForestによるモデル1 (Optunaにより最適化されたハイパーパラメータを使用)
class model_rf1:
    def __init__(self):
        self.model = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        self.model = RandomForestRegressor(max_depth = 29,
                                           n_estimators = 1000,
                                           random_state = 1234)
        self.model.fit(X_train, y_train)
        
    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred
    
# RandomForestによるモデル2    
class model_rf2:
    def __init__(self):
        self.model = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        # params = {max_depth:200, n_estimators:10, random_state:1234}
        self.model = RandomForestRegressor(max_depth =15,
                                           n_estimators = 300,
                                           random_state = 1234)
        self.model.fit(X_train, y_train)
        
    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred

# Ridge回帰によるモデル
# alpha値は0.1~100までの間で0.1刻みでRMSEを計測し、最も当てはまりがよかった数値を使用。
class model_ridge:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        self.model = Ridge(alpha = 12.6)
        self.model.fit(X_train, y_train)
        
    def predict(self, x):
        x = self.scaler.transform(x)
        y_pred = self.model.predict(x)
        return y_pred

# kの数に差をつけたモデルを複数作成し過学習を抑える。
# k近傍法によるモデル1
class model_knr1:
    def __init__(self):
        self.model = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        model = KNeighborsRegressor(n_neighbors = 80)
        model.fit(X_train, y_train)
        self.model = model
        
    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred   

# k近傍法によるモデル2    
class model_knr2:
    def __init__(self):
        self.model = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        model = KNeighborsRegressor(n_neighbors = 35)
        model.fit(X_train, y_train)
        self.model = model
        
    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred

# スタッキング2層目用の線形回帰
class model_lr:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
    def predict(self, x):
        x = self.scaler.transform(x)
        y_pred = self.model.predict(x)
        return y_pred    

