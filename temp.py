#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:32:37 2021

@author: caliariitalo
"""
import pandas as pd
import numpy as np
from myPLS import sgautoPLS
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from myPCA import *
from sklearn import preprocessing

X = pd.read_csv('plums.csv').values[:,1:]

# #Xraw, Yraw = load_boston(return_X_y=True)
# # Get reference values
# Yraw = data['Brix'].values
# # Get spectra
# Xraw = data.drop(['Brix'], axis=1).values
# # Get wavelengths
# wl = np.arange(1100,2300,2)

#X_train, X_test, Y_train, Y_test = train_test_split(Xraw, Yraw)

#bestoutput = sgautoPLS(X_train, Y_train, X_test, Y_test, 5, cv=10)

X = np.log(1.0/X)
wl = np.arange(1100,2300,2)