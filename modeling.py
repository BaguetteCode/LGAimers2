# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:49:56 2023

@author: lsy
"""
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,6:], train["Y_Class"])

lgb = LGBMClassifier(n_estimators = 400)
lgb.fit(X_train, y_train)

lgb_pred = lgb.predict(X_test)


y_pred = lgb.predict(test.iloc[:,4:])

sub = pd.read_csv("sample_submission.csv")
sub["Y_Class"] = y_pred

sub.to_csv("lightGBM_submission.csv", index = False)
