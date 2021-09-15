#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 10:35:49 2021
PCA Routine
@author: caliariitalo
"""
def runPCA(X, prep='mncn'):
    "runPCA(X, prep='mncn')"
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn import preprocessing

    if prep == 'mncn': X = preprocessing.scale(X, with_mean='True', with_std='False')
    if prep == 'auto': X = preprocessing.scale(X, with_mean='True', with_std='True')

    index = np.arange(0, len(X)-1)
    max_components = 10

    pca = PCA(max_components)
    pca.fit(X)

    fig1 = plt.figure(figsize = (5,5), dpi=300)
    pcplot = fig1.add_subplot(1,1,1) 
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

    scores = pca.fit_transform(X)
    loadings = pca.components_

    n_components = 3

    euclidean = np.zeros(X.shape[0])
    for i in range(n_components):
        euclidean += (scores[:,i] - np.mean(scores[:,:n_components]))**2/np.var(scores[:,:n_components])
    colors = [plt.cm.jet(float(i)/max(euclidean)) for i in euclidean]

    fig2 = plt.figure(figsize = (12, 6), dpi=300)
    scoresplot1 = fig2.add_subplot(1,2,1) 
    scoresplot1.scatter(scores[:, 0], scores[:, 1], c=colors, edgecolors='k', s=60)
    scoresplot1.set_xlabel('Scores PC 1 (' + str(pca.explained_variance_ratio_[0]*100) + '%)',
                         fontsize = 10)
    scoresplot1.set_ylabel('Scores PC 2 (' + str(pca.explained_variance_ratio_[1]*100) + '%)', 
                         fontsize = 10)
    scoresplot1.set_xlim(min(min(scores[:, 0]), min(scores[:, 1])),
                       max(max(scores[:, 0]), max(scores[:, 1])))
    scoresplot1.set_ylim(min(min(scores[:, 0]), min(scores[:, 1])),
                       max(max(scores[:, 0]), max(scores[:, 1])))
    for xi, yi, indexi in zip(scores[:, 0], scores[:, 1], index):
        scoresplot1.annotate(str(indexi), xy = (xi, yi))

    loadingsplot1 = fig2.add_subplot(1,2,2) 
    loadingsplot1.plot(loadings[0,:], label='Loadings PC1')
    loadingsplot1.plot(loadings[1,:], label='Loadings PC2')
    loadingsplot1.set_xlabel('Variables')
    loadingsplot1.set_ylabel('Loadings')
    loadingsplot1.legend()
    
    fig3 = plt.figure(figsize = (12,6), dpi=300)
    scoresplot2 = fig3.add_subplot(1,2,1)
    scoresplot2.scatter(scores[:, 1], scores[:, 2], c=colors, edgecolors='k', s=60)
    scoresplot2.set_xlabel('Scores PC 2 (' + str(pca.explained_variance_ratio_[1]*100) + '%)', 
                         fontsize = 10)
    scoresplot2.set_ylabel('Scores PC 3 (' + str(pca.explained_variance_ratio_[2]*100) + '%)',  
                         fontsize = 10)
    scoresplot2.set_xlim(min(min(scores[:, 1]), min(scores[:, 2])),
                       max(max(scores[:, 1]), max(scores[:, 2])))
    scoresplot2.set_ylim(min(min(scores[:, 1]), min(scores[:, 2])),
                       max(max(scores[:, 1]), max(scores[:, 2])))
    for xi, yi, indexi in zip(scores[:, 1], scores[:, 2], index):
        scoresplot2.annotate(str(indexi), xy = (xi, yi))

    loadingsplot2 = fig3.add_subplot(1,2,2) 
    loadingsplot2.plot(loadings[1,:], label='Loadings PC2')
    loadingsplot2.plot(loadings[2,:], label='Loadings PC3')
    loadingsplot2.set_xlabel('Variables') 
    loadingsplot2.set_ylabel('Loadings')
    loadingsplot2.legend()