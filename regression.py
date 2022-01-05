# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 18:16:22 2022

@author: calia
"""
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV



pipe = make_pipeline(outlier removal(), 
                     train_test_split(), 
                     line_preprocessing(), 
                     StandardScaler(), 
                     PLSRegression())


parameters = {'n_components': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
clf = GridSearchCV(estimator=PLSRegression(), para)
clf.fit(X, y)
