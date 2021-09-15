#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usefull Chemometrics routines
@author: caliariitalo
"""
def splisamples(X, Y, TestSplit=0.25):
    "[X_train, Y_train, X_test, Y_test] = splisamples(X, y, TestSplit=0.25)"
    import pandas as pd

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
   
    index = pd.DataFrame(index=Y.sort_values(0).index)
    indextrain = pd.DataFrame()
    indextest = index
   
    X_train = X
    Y_train = Y
    X_test = pd.DataFrame()
    Y_test = pd.DataFrame()

    for temp in range(1, len(index)):
        if len(index) - len(indextest.iloc[int(len(indextest)/temp):
                                           len(indextest)-int(len(indextest)/temp):
                                           int(len(indextest)/temp)]) < len(index)*(1-TestSplit): break

    len(indextest.iloc[int(len(indextest)/temp):
                       len(indextest)-int(len(indextest)/temp):
                       int(len(indextest)/temp)])

    indextest = indextest.iloc[int(len(indextest)/temp):    
                               len(indextest)-int(len(indextest)/temp):
                               int(len(indextest)/temp)]

    indextrain = index.drop(indextest.index)

    X_train = X.iloc[indextrain.index,:]
    Y_train = Y.iloc[indextrain.index]
    X_test = X.iloc[indextest.index,:]
    Y_test = Y.iloc[indextest.index]

    return [X_train, Y_train, X_test, Y_test]

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
    
def runPLS(X_train, Y_train, X_test, Y_test, n_components, prep='mncn', cv=10, plot='off'):
    "output = runPLS(X_train, Y_train, X_test, Y_test, n_components, prep='mncn', cv=10, plot='off')"

    import pandas as pd
    import math
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import mean_squared_error, r2_score

    if prep == 'mncn':
        X_train = X_train - X_train.mean(0)
        X_test = X_test - X_test.mean(0)
#        Y_train = Y_train - Y_train.mean(0)
#        Y_test = Y_test - Y_test.mean(0)
#    if prep == 'auto':
#        X_train = preprocessing.scale(X_train, with_mean='True', with_std='True')
#        X_test = (X_test - X_train.mean(0))/X_train.std(0)

    pls = PLSRegression(n_components)
    pls.fit(X_train, Y_train)

    Y_train_predicted = pls.predict(X_train)
    Y_train_predicted_CV = cross_val_predict(pls, X_train, Y_train, cv=cv)
    Y_test_predicted = pls.predict(X_test)

    RMSEC = math.sqrt(mean_squared_error(Y_train, Y_train_predicted))
    RMSECV = math.sqrt(mean_squared_error(Y_train, Y_train_predicted_CV))
    RMSEP = math.sqrt(mean_squared_error(Y_test, Y_test_predicted))
    R2 = r2_score(Y_train, Y_train_predicted)
    R2CV = r2_score(Y_train, Y_train_predicted_CV)
    R2P = r2_score(Y_test, Y_test_predicted)

    if plot == 'on':
        fig = plt.figure(figsize = (5,5), dpi=300)
        mxp = fig.add_subplot(1,1,1) 
        mxp.scatter(Y_train, Y_train_predicted_CV, label = 'Train set')
        mxp.scatter(Y_test, Y_test_predicted, label = 'Test set')
        mxp.set_xlabel('Measured')
        mxp.set_ylabel('Predicted')
        mxp.set_xlim(min(min(Y_train), 
                         min(Y_train_predicted_CV), 
                         min(Y_test), 
                         min(Y_test_predicted)),
                     max(max(Y_train), 
                         max(Y_train_predicted_CV), 
                         max(Y_test), 
                         max(Y_test_predicted))
                     )
        mxp.set_ylim(min(min(Y_train), 
                         min(Y_train_predicted_CV), 
                         min(Y_test), 
                         min(Y_test_predicted)),
                     max(max(Y_train), 
                         max(Y_train_predicted_CV), 
                         max(Y_test), 
                         max(Y_test_predicted))
                     )
        mxp.legend()

    output = pd.DataFrame({'Components': [n_components],
                           'RMSEC': [RMSEC],
                           'R2': [R2],
                           'RMSECV': [RMSECV],
                           'R2CV': [R2CV],
                           'RMSEP': [RMSEP],
                           'R2P': [R2P]
                           })

    return output

def optPLS(X_train, Y_train, X_test, Y_test, max_components, prep='mncn', cv=10, plot='off'):
    "output = optPLS(X_train, Y_train, X_test, Y_test, max_components, cv=10, plot='off')"

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    output = pd.DataFrame()

    for n_components in range(1, max_components+1):
        output = output.append(runPLS(X_train, Y_train, X_test, Y_test,
                                      n_components, prep=prep, cv=cv, plot='off'))

    if plot == 'on':
        fig1 = plt.figure(figsize = (5,5), dpi=300)
        RMSEplot = fig1.add_subplot(1,1,1) 
        LV = np.arange(1, max_components+1)
        RMSEplot.plot(LV, output.RMSEC, label = 'RMSEC')
        RMSEplot.plot(LV, output.RMSECV, label = 'RMSECV')
        RMSEplot.plot(LV, output.RMSEP, label = 'RMSEP')
        RMSEplot.set_xlabel('LV', fontsize = 10)
        RMSEplot.set_ylabel('A. U.', fontsize = 10)
        RMSEplot.legend()

        fig2 = plt.figure(figsize = (5,5), dpi=300)
        R2plot = fig2.add_subplot(1,1,1)
        LV = np.arange(1, max_components+1)
        R2plot.plot(LV, output.R2, label = 'R2')
        R2plot.plot(LV, output.R2CV, label = 'R2CV')
        R2plot.plot(LV, output.R2P, label = 'R2P')
        R2plot.set_xlabel('LV', fontsize = 10)
        R2plot.set_ylabel('A. U.', fontsize = 10)
        R2plot.legend()

    return output

def autoPLS(X_train, Y_train, X_test, Y_test, max_components, prep='mncn', cv=10, plot='off'):
    "bestoutput = autoPLS(X_train, Y_train, X_test, Y_test, max_components, cv=10, plot='off')"

    import numpy as np

    output = optPLS(X_train, Y_train, X_test, Y_test,
                    max_components, prep=prep, cv=cv, plot=plot)

    diff = np.diff(output.RMSECV)

    for n_components in range(1,len(diff)+1):
        if diff[n_components-1]/output.RMSECV.iloc[n_components-1] > -0.1: break

    bestoutput = runPLS(X_train, Y_train, X_test, Y_test,
                        n_components, prep=prep, cv=cv, plot=plot)

    return bestoutput

def sgPLS(X_train, Y_train, X_test, Y_test, max_components, window_length, polyorder, deriv, prep='mncn', cv=10, plot='off'):
    "bestoutput = sgPLS(X_train, Y_train, X_test, Y_test, max_components, window_length, polyorder, deriv, cv=10, plot='off')"

    from scipy.signal import savgol_filter

    X_train_sg = savgol_filter(X_train, window_length, polyorder, deriv)
    X_test_sg = savgol_filter(X_test, window_length, polyorder, deriv)

    bestoutput = autoPLS(X_train_sg, Y_train, X_test_sg, Y_test,
                         max_components, prep=prep, cv=cv, plot=plot)

    return bestoutput

def sgautoPLS(X_train, Y_train, X_test, Y_test, max_components, prep='mncn', cv=10):
    "bestoutput = sgautoPLS(X_train, Y_train, X_test, Y_test, max_components, cv=10)"
    import pandas as pd

    sg = pd.DataFrame()
    output = pd.DataFrame()
    bestoutput = pd.DataFrame()
    run = 0

    for window_length in range(5, int(len(X_test.T)*.15), 2):
        for deriv in range(0, 3):
            for polyorder in range(deriv, 3):
                run = run + 1
                sg = sg.append((pd.DataFrame({'Run': [run],
                                              'Window': [window_length],
                                              'PolyOrder': [polyorder],
                                              'Derivative': [deriv]
                                              })))
                output = sgPLS(X_train, Y_train, X_test, Y_test,
                               max_components,
                               window_length, polyorder, deriv,
                               prep=prep, cv=cv, plot='off')
                output['Run'] = run
                bestoutput = bestoutput.append(output)

    sg = sg.set_index('Run')
    bestoutput = bestoutput.set_index('Run')
    bestoutput = bestoutput.join(sg)

    return bestoutput