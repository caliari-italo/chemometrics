#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:32:37 2021

@author: caliariitalo
"""
from myChemometrics import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from statistics import variance
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.cross_decomposition import PLSRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

[output, vs] = varsel(X_train, y_train, X_test, y_test, info='all', max_components=15, cv=10, plot='off')
