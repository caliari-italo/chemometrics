#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 10:35:49 2021
PCA Routine
@author: caliariitalo
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv("peach_spectra_brix.csv")
Xraw = data.iloc[:, 1::]
index = np.arange(0, len(Xraw)-1)

# fig = plt.figure(figsize = (10,10))
# dataplot = fig.add_subplot(1,1,1) 
# dataplot.plot(Xraw.T)
# dataplot.set_title('Raw Data', fontsize = 15)
# dataplot.set_xlabel('Abscissa Axys', fontsize = 10)
# dataplot.set_ylabel('Ordinate Axys', fontsize = 10)

#Preprocessing
X = Xraw - Xraw.mean()

# fig = plt.figure(figsize = (10,10))
# dataplot = fig.add_subplot(1,1,1) 
# dataplot.plot(X.T)
# dataplot.set_title('Transformed Data', fontsize = 15)
# dataplot.set_xlabel('Abscissa Axys', fontsize = 10)
# dataplot.set_ylabel('Ordinate Axys', fontsize = 10)

max_components = 10

pca = PCA(max_components)
pca.fit(X)

fig = plt.figure(figsize = (5,5))
pcplot = fig.add_subplot(1,1,1) 
pcplot.plot(np.arange(1, max_components+1), 
            pca.explained_variance_ratio_*100,
            label = 'Captured variance (Individual)')
pcplot.plot(np.arange(1, max_components+1),
            [sum(pca.explained_variance_ratio_[0:temp+1]*100) 
             for temp in range(max_components)],
            label = 'Captured variance (Total)')
pcplot.set_xlabel('PCs', fontsize = 10)
pcplot.set_ylabel('Captured Variance (%)', fontsize = 10)
pcplot.legend()

n_components = 3

pca = PCA(n_components)
pca.fit(X)
scores = pca.fit_transform(X)

fig = plt.figure(figsize = (10,5))
scoreplot = fig.add_subplot(1,2,1) 
scoreplot.scatter(scores[:, 0], scores[:, 1])
scoreplot.set_xlabel('PC 1 (' + str(pca.explained_variance_ratio_[0]*100) + '%)',
                     fontsize = 10)
scoreplot.set_ylabel('PC 2 (' + str(pca.explained_variance_ratio_[1]*100) + '%)', 
                     fontsize = 10)
for xi, yi, indexi in zip(scores[:, 0], scores[:, 1], index):
    scoreplot.annotate(str(indexi), xy = (xi, yi))
    
loadings = pca.components_
    
Xrec = pd.DataFrame(np.matmul(scores[:, 0:n_components-1], loadings[0:n_components-1, :]),
                   columns = X.columns) + Xraw.mean()

scoreplot = fig.add_subplot(1,2,2) 
scoreplot.scatter(scores[:, 1], scores[:, 2])
scoreplot.set_xlabel('PC 2 (' + str(pca.explained_variance_ratio_[1]*100) + '%)', 
                     fontsize = 10)
scoreplot.set_ylabel('PC 3 (' + str(pca.explained_variance_ratio_[2]*100) + '%)',  
                     fontsize = 10)
for xi, yi, indexi in zip(scores[:, 1], scores[:, 2], index):
    scoreplot.annotate(str(indexi), xy = (xi, yi))    
    
del(xi, yi, indexi, fig, index, max_components)