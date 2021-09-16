#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:32:37 2021

@author: caliariitalo
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from myChemometrics import *

from sklearn import preprocessing

data = pd.read_csv('peach_spectra_brix.csv')
# Get reference values
Y = data['Brix'].values
# Get spectra
X = data.drop(['Brix'], axis=1).values

[X_train, Y_train, X_test, Y_test] = splitsamples(X, Y, TestSplit=0.25)

bestoutput = sgautoPLS(X_train, Y_train, X_test, Y_test, 10)

output = sgPLS(X_train, Y_train, X_test, Y_test, 10,
               int(bestoutput.iloc[0,:].Window),
               int(bestoutput.iloc[0,:].PolyOrder),
               int(bestoutput.iloc[0,:].Derivative),
               cv=10, plot='on')

runPCA(savgol_filter(X_train,
                     int(bestoutput.iloc[0,:].Window),
                     int(bestoutput.iloc[0,:].PolyOrder),
                     int(bestoutput.iloc[0,:].Derivative)))

runPCA2(savgol_filter(X_train,
                     int(bestoutput.iloc[0,:].Window),
                     int(bestoutput.iloc[0,:].PolyOrder),
                     int(bestoutput.iloc[0,:].Derivative)),
        savgol_filter(X_test,
                     int(bestoutput.iloc[0,:].Window),
                     int(bestoutput.iloc[0,:].PolyOrder),
                     int(bestoutput.iloc[0,:].Derivative)))
