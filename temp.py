#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:32:37 2021

@author: caliariitalo
"""

from myPLS import autoPLS
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


Xraw, Yraw = load_boston(return_X_y=True)

X_train, X_test, Y_train, Y_test = train_test_split(Xraw, Yraw)

bestoutput = autoPLS(X_train, Y_train, X_test, Y_test, 15, cv=10, plot='on')
