#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:32:37 2021

@author: caliariitalo
"""
from myChemometrics import *
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

[X_train, y_train, X_test, y_test] = splitsamples(X, y)

autosgPLS = autosgPLS(X_train, y_train, X_test, y_test, max_components=15, cv=10, plot='on')

X_train_sg = pd.DataFrame(savgol_filter(X_train, int(autosgPLS['Window']), int(autosgPLS['PolyOrder']), int(autosgPLS['Derivative'])))
X_test_sg = pd.DataFrame(savgol_filter(X_test, int(autosgPLS['Window']), int(autosgPLS['PolyOrder']), int(autosgPLS['Derivative'])))

autoPLS = autoPLS(X_train_sg, y_train, X_test_sg, y_test, max_components=15, cv=10, plot='on')
