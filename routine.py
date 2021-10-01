#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:32:37 2021

@author: caliariitalo
"""
from chemometrics import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from statistics import variance
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.cross_decomposition import PLSRegression

X = (np.array(X)).squeeze()
y = (np.array(y)).squeeze()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

[output0, vs0] = varsel(X_train, y_train, X_test, y_test, info='all', step=0.05, max_components=10, cv=10, plot='on')

plt.plot(vector)



[output1, vs1] = varsel(X_train, y_train, X_test, y_test, info='PLSReg', step=0.05, max_components=10, cv=10, plot='on')
[output2, vs2] = varsel(X_train, y_train, X_test, y_test, info='Loadings', step=0.05, max_components=10, cv=10, plot='on')
[output3, vs3] = varsel(X_train, y_train, X_test, y_test, info='Weigths', step=0.05, max_components=10, cv=10, plot='on')
[output4, vs4] = varsel(X_train, y_train, X_test, y_test, info='std', step=0.05, max_components=10, cv=10, plot='on')
[output5, vs5] = varsel(X_train, y_train, X_test, y_test, info='SQR', step=0.05, max_components=10, cv=10, plot='on')
[output6, vs6] = varsel(X_train, y_train, X_test, y_test, info='LinReg', step=0.05, max_components=10, cv=10, plot='on')
[output7, vs7] = varsel(X_train, y_train, X_test, y_test, info='all', step=0.05, max_components=10, cv=10, plot='on')
[output8, vs8] = varsel(X_train, y_train, X_test, y_test, info='all', step=0.05, max_components=10, cv=10, plot='on')
