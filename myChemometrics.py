#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usefull Chemometrics routines
@author: caliariitalo
"""
def splitsamples(X, Y, TestSplit=0.25):
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

def runPCA(X, prep='none'):
    "runPCA(X, prep='none')"

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
    pcplot.set_xlabel('PCs', fontsize = 10)
    pcplot.set_ylabel('Captured Variance (%)', fontsize = 10)
    pcplot.plot(np.arange(1, max_components+1), 
                pca.explained_variance_ratio_*100,
                label = 'Captured variance (Individual)')
    pcplot.plot(np.arange(1, max_components+1),
                [sum(pca.explained_variance_ratio_[0:temp+1]*100) 
                 for temp in range(max_components)],
                label = 'Captured variance (Total)')
    pcplot.legend()

    scores = pca.transform(X)
    loadings = pca.components_

    n_components = 3

    euclidean = np.zeros(X.shape[0])
    for i in range(n_components):
        euclidean += (scores[:,i] - np.mean(scores[:,:n_components]))**2/np.var(scores[:,:n_components])
    colors = [plt.cm.jet(float(i)/max(euclidean)) for i in euclidean]

    fig2 = plt.figure(figsize = (12, 6), dpi=300)
    scoresplot1 = fig2.add_subplot(1,2,1) 
    scoresplot1.set_xlabel('Scores PC 1 (' + str(pca.explained_variance_ratio_[0]*100) + '%)',
                         fontsize = 10)
    scoresplot1.set_ylabel('Scores PC 2 (' + str(pca.explained_variance_ratio_[1]*100) + '%)', 
                         fontsize = 10)
    scoresplot1.set_xlim(min(min(scores[:, 0]), min(scores[:, 1])),
                       max(max(scores[:, 0]), max(scores[:, 1])))
    scoresplot1.set_ylim(min(min(scores[:, 0]), min(scores[:, 1])),
                       max(max(scores[:, 0]), max(scores[:, 1])))
    scoresplot1.scatter(scores[:, 0], scores[:, 1], c=colors, edgecolors='k', s=60)
    for xi, yi, indexi in zip(scores[:, 0], scores[:, 1], index):
        scoresplot1.annotate(str(indexi), xy = (xi, yi))

    loadingsplot1 = fig2.add_subplot(1,2,2)
    loadingsplot1.set_xlabel('Variables')
    loadingsplot1.set_ylabel('Loadings')
    loadingsplot1.plot(loadings[0,:], label='Loadings PC1')
    loadingsplot1.plot(loadings[1,:], label='Loadings PC2')
    loadingsplot1.legend()
    
    fig3 = plt.figure(figsize = (12,6), dpi=300)
    scoresplot2 = fig3.add_subplot(1,2,1)
    scoresplot2.set_xlabel('Scores PC 2 (' + str(pca.explained_variance_ratio_[1]*100) + '%)', 
                         fontsize = 10)
    scoresplot2.set_ylabel('Scores PC 3 (' + str(pca.explained_variance_ratio_[2]*100) + '%)',  
                         fontsize = 10)
    scoresplot2.set_xlim(min(min(scores[:, 1]), min(scores[:, 2])),
                       max(max(scores[:, 1]), max(scores[:, 2])))
    scoresplot2.set_ylim(min(min(scores[:, 1]), min(scores[:, 2])),
                       max(max(scores[:, 1]), max(scores[:, 2])))
    scoresplot2.scatter(scores[:, 1], scores[:, 2], c=colors, edgecolors='k', s=60)
    for xi, yi, indexi in zip(scores[:, 1], scores[:, 2], index):
        scoresplot2.annotate(str(indexi), xy = (xi, yi))

    loadingsplot2 = fig3.add_subplot(1,2,2) 
    loadingsplot2.set_xlabel('Variables') 
    loadingsplot2.set_ylabel('Loadings')
    loadingsplot2.plot(loadings[1,:], label='Loadings PC2')
    loadingsplot2.plot(loadings[2,:], label='Loadings PC3')
    loadingsplot2.legend()

def runPCA2(X1, X2, prep='none'):
    "runPCA2(X, X2, prep='none')"

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn import preprocessing

    if prep == 'mncn': X1 = preprocessing.scale(X1, with_mean='True', with_std='False')
    if prep == 'auto': X1 = preprocessing.scale(X1, with_mean='True', with_std='True')

    index1 = np.arange(0, len(X1)-1)
    index2 = np.arange(0, len(X2)-1)

    max_components = 10

    pca = PCA(max_components)
    pca.fit(X1)

    fig1 = plt.figure(figsize = (5,5), dpi=300)
    pcplot = fig1.add_subplot(1,1,1) 
    pcplot.set_xlabel('PCs', fontsize = 10)
    pcplot.set_ylabel('Captured Variance (%)', fontsize = 10)
    pcplot.plot(np.arange(1, max_components+1), 
                pca.explained_variance_ratio_*100,
                label = 'Captured variance (Individual)')
    pcplot.plot(np.arange(1, max_components+1),
                [sum(pca.explained_variance_ratio_[0:temp+1]*100) 
                 for temp in range(max_components)],
                label = 'Captured variance (Total)')
    pcplot.legend()

    scores1 = pca.transform(X1)
    scores2 = pca.transform(X2)
    loadings = pca.components_

    fig2 = plt.figure(figsize = (12, 6), dpi=300)
    scoresplot1 = fig2.add_subplot(1,2,1)
    scoresplot1.set_xlabel('Scores PC 1 (' + str(pca.explained_variance_ratio_[0]*100) + '%)',
                         fontsize = 10)
    scoresplot1.set_ylabel('Scores PC 2 (' + str(pca.explained_variance_ratio_[1]*100) + '%)', 
                         fontsize = 10)
    scoresplot1.set_xlim(min(min(scores1[:, 0]), min(scores1[:, 1]),
                             min(scores2[:, 0]), min(scores2[:, 1])),
                         max(max(scores1[:, 0]), max(scores1[:, 1]),
                             max(scores2[:, 0]), max(scores2[:, 1])))
    scoresplot1.set_ylim(min(min(scores1[:, 0]), min(scores1[:, 1]),
                             min(scores2[:, 0]), min(scores2[:, 1])),
                         max(max(scores1[:, 0]), max(scores1[:, 1]),
                             max(scores2[:, 0]), max(scores2[:, 1])))
    scoresplot1.scatter(scores1[:, 0], scores1[:, 1], label='X1')
    for xi, yi, indexi in zip(scores1[:, 0], scores1[:, 1], index1):
        scoresplot1.annotate(str(indexi), xy = (xi, yi))
    scoresplot1.scatter(scores2[:, 0], scores2[:, 1], label='X2')
    for xi, yi, indexi in zip(scores2[:, 0], scores2[:, 1], index2):
        scoresplot1.annotate(str(indexi), xy = (xi, yi))
    scoresplot1.legend()

    loadingsplot1 = fig2.add_subplot(1,2,2)
    loadingsplot1.set_xlabel('Variables')
    loadingsplot1.set_ylabel('Loadings')
    loadingsplot1.plot(loadings[0,:], label='Loadings PC1')
    loadingsplot1.plot(loadings[1,:], label='Loadings PC2')

    loadingsplot1.legend()
    
    fig3 = plt.figure(figsize = (12,6), dpi=300)
    scoresplot2 = fig3.add_subplot(1,2,1)
    scoresplot2.set_xlabel('Scores PC 2 (' + str(pca.explained_variance_ratio_[1]*100) + '%)', 
                         fontsize = 10)
    scoresplot2.set_ylabel('Scores PC 3 (' + str(pca.explained_variance_ratio_[2]*100) + '%)',  
                         fontsize = 10)
    scoresplot2.set_xlim(min(min(scores1[:, 1]), min(scores1[:, 2]),
                             min(scores2[:, 1]), min(scores2[:, 2])),
                         max(max(scores1[:, 1]), max(scores1[:, 2]),
                             max(scores2[:, 1]), max(scores2[:, 2])))
    scoresplot2.set_ylim(min(min(scores1[:, 1]), min(scores1[:, 2]),
                             min(scores2[:, 1]), min(scores2[:, 2])),
                         max(max(scores1[:, 1]), max(scores1[:, 2]),
                             max(scores2[:, 1]), max(scores2[:, 2])))
    scoresplot2.scatter(scores1[:, 1], scores1[:, 2], label='X1')
    for xi, yi, indexi in zip(scores1[:, 1], scores1[:, 2], index1):
        scoresplot2.annotate(str(indexi), xy = (xi, yi))
    scoresplot2.scatter(scores2[:, 1], scores2[:, 2], label='X2')
    for xi, yi, indexi in zip(scores2[:, 1], scores2[:, 2], index2):
        scoresplot2.annotate(str(indexi), xy = (xi, yi))
    scoresplot2.legend()

    loadingsplot2 = fig3.add_subplot(1,2,2)
    loadingsplot2.set_xlabel('Variables')
    loadingsplot2.set_ylabel('Loadings')
    loadingsplot2.plot(loadings[1,:], label='Loadings PC2')
    loadingsplot2.plot(loadings[2,:], label='Loadings PC3')
    loadingsplot2.legend()

def runPLS(X_train, Y_train, X_test, Y_test, n_components, prep='none', cv=10, plot='off'):
    "output = runPLS(X_train, Y_train, X_test, Y_test, n_components, prep='none', cv=10, plot='off')"

    import pandas as pd
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import mean_squared_error, r2_score

    if prep == 'mncn':
        X_train = X_train - X_train.mean(0)
        X_test = X_test - X_train.mean(0)

    pls = PLSRegression(n_components)
    pls.fit(X_train, Y_train)

    Y_train_predicted = pls.predict(X_train)
    Y_train_predicted_CV = cross_val_predict(pls, X_train, Y_train, cv=cv)
    Y_test_predicted = pls.predict(X_test)

    RMSEC = math.sqrt(mean_squared_error(Y_train, Y_train_predicted))
    RMSECV = math.sqrt(mean_squared_error(Y_train, Y_train_predicted_CV))
    RMSEP = math.sqrt(mean_squared_error(Y_test, Y_test_predicted))
    R2C = r2_score(Y_train, Y_train_predicted)
    R2CV = r2_score(Y_train, Y_train_predicted_CV)
    R2P = r2_score(Y_test, Y_test_predicted)

    if plot == 'on':
        fig = plt.figure(figsize = (5,5), dpi=300)
        mxp = fig.add_subplot(1,1,1) 
        mxp.set_xlabel('Measured')
        mxp.set_ylabel('Predicted')
        mxp.set_xlim(min(min(np.array(Y_train)),
                         min(Y_train_predicted_CV),
                         min(np.array(Y_test)),
                         min(Y_test_predicted)),
                     max(max(np.array(Y_train)),
                         max(Y_train_predicted_CV),
                         max(np.array(Y_test)),
                         max(Y_test_predicted)))
        mxp.set_ylim(min(min(np.array(Y_train)),
                         min(Y_train_predicted_CV),
                         min(np.array(Y_test)),
                         min(Y_test_predicted)),
                     max(max(np.array(Y_train)),
                         max(Y_train_predicted_CV),
                         max(np.array(Y_test)),
                         max(Y_test_predicted)))
        mxp.scatter(Y_train, Y_train_predicted_CV, label = 'Train set')
        mxp.scatter(Y_test, Y_test_predicted, label = 'Test set')
        index1 = np.arange(0, len(Y_train))
        index2 = np.arange(0, len(Y_test))
        for xi, yi, indexi in zip(np.array(Y_train), np.array(Y_train_predicted_CV), index1):
            mxp.annotate(str(indexi), xy = (xi, yi))
        for xi, yi, indexi in zip(np.array(Y_test), np.array(Y_test_predicted), index2):
            mxp.annotate(str(indexi), xy = (xi, yi))
        mxp.legend()

    output = pd.DataFrame({'Components': [n_components],
                           'RMSEC': [RMSEC],
                           'R2C': [R2C],
                           'RMSECV': [RMSECV],
                           'R2CV': [R2CV],
                           'RMSEP': [RMSEP],
                           'R2P': [R2P]
                           })

    return output

def autoPLS(X_train, Y_train, X_test, Y_test, max_components=20, prep='none', cv=10, plot='off'):
    "output = autoPLS(X_train, Y_train, X_test, Y_test, max_components=20, cv=10, plot='off')"

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_predict

    output = pd.DataFrame()

    for n_components in range(1, max_components+1):
        output = output.append(runPLS(X_train, Y_train, X_test, Y_test,
                                      n_components, prep=prep, cv=cv, plot='off'))
    diff = np.diff(output.RMSECV)

    for n_components in range(1,len(diff)+1):
        if diff[n_components-1]/output.RMSECV.iloc[n_components-1] > -0.1: break

    if plot == 'on':
        fig1 = plt.figure(figsize = (10, 5), dpi=300)
        RMSEplot = fig1.add_subplot(1, 2, 1) 
        RMSEplot.plot(output['Components'], output['RMSEC'], label = 'RMSEC')
        RMSEplot.plot(output['Components'], output['RMSECV'], label = 'RMSECV')
        RMSEplot.plot(output['Components'], output['RMSEP'], label = 'RMSEP')
        RMSEplot.set_xlabel('Components', fontsize = 10)
        RMSEplot.set_ylabel('A. U.', fontsize = 10)
        RMSEplot.legend()

        R2plot = fig1.add_subplot(1, 2, 2)
        R2plot.plot(output['Components'], output['R2C'], label = 'R2C')
        R2plot.plot(output['Components'], output['R2CV'], label = 'R2CV')
        R2plot.plot(output['Components'], output['R2P'], label = 'R2P')
        R2plot.set_xlabel('Components', fontsize = 10)
        R2plot.set_ylabel('A. U.', fontsize = 10)
        R2plot.legend()

    output = output.sort_values(by=['RMSECV'])

    if plot == 'on':
        RMSEplot.plot(output['Components'].iloc[0], output['RMSEC'].iloc[0], 'P', ms=10, mfc='red')
        RMSEplot.plot(output['Components'].iloc[0], output['RMSECV'].iloc[0], 'P', ms=10, mfc='red')
        RMSEplot.plot(output['Components'].iloc[0], output['RMSEP'].iloc[0], 'P', ms=10, mfc='red')
        R2plot.plot(output['Components'].iloc[0], output['R2C'].iloc[0], 'P', ms=10, mfc='red')
        R2plot.plot(output['Components'].iloc[0], output['R2CV'].iloc[0], 'P', ms=10, mfc='red')
        R2plot.plot(output['Components'].iloc[0], output['R2P'].iloc[0], 'P', ms=10, mfc='red')

        if prep == 'mncn':
            X_train = X_train - X_train.mean(0)
            X_test = X_test - X_train.mean(0)

        pls = PLSRegression(output['Components'].iloc[0])
        pls.fit(X_train, Y_train)

        Y_train_predicted_CV = cross_val_predict(pls, X_train, Y_train, cv=cv)
        Y_test_predicted = pls.predict(X_test)
        
        fig3 = plt.figure(figsize = (5,5), dpi=300)
        mxp = fig3.add_subplot(1,1,1) 
        mxp.set_xlabel('Measured')
        mxp.set_ylabel('Predicted')
        mxp.set_xlim(min(min(np.array(Y_train)),
                         min(Y_train_predicted_CV),
                         min(np.array(Y_test)),
                         min(Y_test_predicted)),
                     max(max(np.array(Y_train)),
                         max(Y_train_predicted_CV),
                         max(np.array(Y_test)),
                         max(Y_test_predicted)))
        mxp.set_ylim(min(min(np.array(Y_train)),
                         min(Y_train_predicted_CV),
                         min(np.array(Y_test)),
                         min(Y_test_predicted)),
                     max(max(np.array(Y_train)),
                         max(Y_train_predicted_CV),
                         max(np.array(Y_test)),
                         max(Y_test_predicted)))
        mxp.scatter(Y_train, Y_train_predicted_CV, label = 'Train set')
        mxp.scatter(Y_test, Y_test_predicted, label = 'Test set')
        index1 = np.arange(0, len(Y_train))
        index2 = np.arange(0, len(Y_test))
        for xi, yi, indexi in zip(np.array(Y_train), np.array(Y_train_predicted_CV), index1):
            mxp.annotate(str(indexi), xy = (xi, yi))
        for xi, yi, indexi in zip(np.array(Y_test), np.array(Y_test_predicted), index2):
            mxp.annotate(str(indexi), xy = (xi, yi))
        mxp.legend()

    return output

def sgPLS(X_train, Y_train, X_test, Y_test, max_components, window_length, polyorder, deriv, prep='none', cv=10, plot='off'):
    "output = sgPLS(X_train, Y_train, X_test, Y_test, max_components, window_length, polyorder, deriv, cv=10, plot='off')"

    from scipy.signal import savgol_filter

    X_train_sg = savgol_filter(X_train, window_length, polyorder, deriv)
    X_test_sg = savgol_filter(X_test, window_length, polyorder, deriv)

    output = autoPLS(X_train_sg, Y_train, X_test_sg, Y_test,
                         max_components, prep=prep, cv=cv, plot=plot)

    return output

def autosgPLS(X_train, Y_train, X_test, Y_test, max_components=20, prep='none', cv=10, plot='off'):
    "output = autosgPLS(X_train, Y_train, X_test, Y_test, max_components=20, prep='none', cv=10, plot='off')"
    import pandas as pd
    from scipy.signal import savgol_filter

    sg = pd.DataFrame()
    temp = pd.DataFrame()
    output = pd.DataFrame()

    for window_length in range(5, int(len(X_test.T)*.15), 2):
        for deriv in range(0, 3):
            for polyorder in range(deriv, 3):
                sg = sg.append((pd.DataFrame({'Window': [window_length],
                                              'PolyOrder': [polyorder],
                                              'Derivative': [deriv]
                                              })))
                temp = sgPLS(X_train, Y_train, X_test, Y_test,
                               max_components,
                               window_length, polyorder, deriv,
                               prep=prep, cv=cv, plot='off')
                output = output.append(temp)

    output = output.join(sg)

    output = output.sort_values(by=['RMSECV'])

    if plot == 'on':
        X_train = savgol_filter(X_train,
                                int(output['Window'].iloc[0]),
                                int(output['PolyOrder'].iloc[0]),
                                int(output['Derivative'].iloc[0]))
        X_test = savgol_filter(X_test,
                                int(output['Window'].iloc[0]),
                                int(output['PolyOrder'].iloc[0]),
                                int(output['Derivative'].iloc[0]))
        if prep == 'mncn':
            X_train = X_train - X_train.mean(0)
            X_test = X_test - X_train.mean(0)

        autoPLS(X_train, Y_train, X_test, Y_test, max_components=max_components, prep=prep, cv=cv, plot='on')

    return output

def varsel(X_train, Y_train, X_test, Y_test, max_components, estimator, method, scoring='neg_mean_squared_error', prep='none', cv=10):
    ""

    from sklearn.feature_selection import RFECV

    if estimator == 'pls':
        autoPLS(X_train, Y_train, X_test, Y_test, max_components=10)
        pls.fit(X_train, Y_train)


    rfecv = RFECV(estimator=estimator, scoring=scoring)
    rfecv.fit(X_train, Y_train)
    X_train = rfecv.fit_transform(X_train, Y_train)
    
    varsel = rfecv.support_

    return varsel