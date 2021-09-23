#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:32:37 2021

@author: caliariitalo
"""
from myChemometrics import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

[X, y] = make_regression(n_samples=200, n_features=500, n_informative=250, effective_rank=5)

X=pd.DataFrame(X)
y=pd.DataFrame(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

[output, varsel] = varsel(X_train, y_train, X_test, y_test, max_components=10, estimator='all', cv=10, plot='on')



output = autoPLS(X_train, y_train, X_test, y_test, max_components=20, cv=10, plot='on')

output = runPLS(X_train, y_train, X_test, y_test, n_components=5, cv=10, plot='on')
