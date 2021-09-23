#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usefull Chemometrics routines
@author: caliariitalo
"""
def splitsamples(X, y, TestSplit=0.25):
    "[X_train, y_train, X_test, y_test] = splisamples(X, y, TestSplit=0.25)"

    import pandas as pd

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    index = pd.DataFrame(index=y.sort_values(0).index)
    indextrain = pd.DataFrame()
    indextest = index

    X_train = X
    y_train = y
    X_test = pd.DataFrame()
    y_test = pd.DataFrame()

    for temp in range(1, len(index)+1):
        if len(index) - len(indextest.iloc[int(len(indextest)/temp):
                                           len(indextest)-int(len(indextest)/temp):
                                           int(len(indextest)/temp)]) < len(index)*(1-TestSplit): break

    indextest = indextest.iloc[int(len(indextest)/temp):
                               len(indextest)-int(len(indextest)/temp):
                               int(len(indextest)/temp)]

    indextrain = index.drop(indextest.index)

    X_train = X.iloc[indextrain.index,:]
    y_train = y.iloc[indextrain.index]
    X_test = X.iloc[indextest.index,:]
    y_test = y.iloc[indextest.index]

    return [X_train, y_train, X_test, y_test]

def runPCA(X):
    "runPCA(X)"

    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    index = np.arange(len(X))
    max_components = 10

    pca = PCA(max_components)
    pca.fit(X)

    fig1 = plt.figure(figsize = (6,6), dpi=300)
    pcplot = fig1.add_subplot(1,1,1) 
    pcplot.set_xlabel('PCs')
    pcplot.set_ylabel('Captured Variance (%)')
    pcplot.plot(np.arange(max_components)+1, 
                pca.explained_variance_ratio_*100,
                label = 'Captured variance (Individual)')
    pcplot.plot(np.arange(max_components)+1,
                [sum(pca.explained_variance_ratio_[0:temp+1]*100)
                 for temp in range(max_components)],
                label = 'Captured variance (Total)')
    pcplot.set_xlim(1, max_components)
    pcplot.legend()

    scores = pca.transform(X)
    loadings = pca.components_

    n_components = 3

    euclidean = np.zeros(len(X))
    for i in range(n_components):
        euclidean += (scores[:,i] - np.mean(scores[:,:n_components]))**2/np.var(scores[:,:n_components])
    colors = [plt.cm.jet(float(i)/max(euclidean)) for i in euclidean]

    fig2 = plt.figure(figsize = (12,6), dpi=300)
    scoresplot1 = fig2.add_subplot(1,2,1) 
    scoresplot1.set_xlabel('Scores PC 1 (' + str(pca.explained_variance_ratio_[0]*100) + '%)')
    scoresplot1.set_ylabel('Scores PC 2 (' + str(pca.explained_variance_ratio_[1]*100) + '%)')
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
    loadingsplot1.plot(np.arange(X.shape[1]), np.zeros(X.shape[1]),'r--')
    loadingsplot1.set_xlim(0, X.shape[1])
    loadingsplot1.legend()
    
    fig3 = plt.figure(figsize = (12,6), dpi=300)
    scoresplot2 = fig3.add_subplot(1,2,1)
    scoresplot2.set_xlabel('Scores PC 2 (' + str(pca.explained_variance_ratio_[1]*100) + '%)')
    scoresplot2.set_ylabel('Scores PC 3 (' + str(pca.explained_variance_ratio_[2]*100) + '%)')
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
    loadingsplot2.plot(np.arange(X.shape[1]), np.zeros(X.shape[1]),'r--')
    loadingsplot2.set_xlim(0, X.shape[1])
    loadingsplot2.legend()

def runPCA2(X1, X2):
    "runPCA2(X1, X2)"

    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    index1 = np.arange(len(X1))
    index2 = np.arange(len(X2))

    max_components = 10

    pca = PCA(max_components)
    pca.fit(X1)

    fig1 = plt.figure(figsize = (6,6), dpi=300)
    pcplot = fig1.add_subplot(1,1,1) 
    pcplot.set_xlabel('PCs')
    pcplot.set_ylabel('Captured Variance (%)')
    pcplot.plot(np.arange(max_components)+1, 
                pca.explained_variance_ratio_*100,
                label = 'Captured variance (Individual)')
    pcplot.plot(np.arange(max_components)+1,
                [sum(pca.explained_variance_ratio_[0:temp+1]*100)
                 for temp in range(max_components)],
                label = 'Captured variance (Total)')
    pcplot.set_xlim(1, max_components)
    pcplot.set_xticks(np.arange(max_components)+1)
    pcplot.legend()

    scores1 = pca.transform(X1)
    scores2 = pca.transform(X2)
    loadings = pca.components_

    fig2 = plt.figure(figsize = (12, 6), dpi=300)
    scoresplot1 = fig2.add_subplot(1,2,1)
    scoresplot1.set_xlabel('Scores PC 1 (' + str(pca.explained_variance_ratio_[0]*100) + '%)')
    scoresplot1.set_ylabel('Scores PC 2 (' + str(pca.explained_variance_ratio_[1]*100) + '%)')
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
    loadingsplot1.plot(np.arange(X1.shape[1]), np.zeros(X1.shape[1]),'r--')
    loadingsplot1.set_xlim(0, X1.shape[1])
    loadingsplot1.legend()

    loadingsplot1.legend()

    fig3 = plt.figure(figsize = (12,6), dpi=300)
    scoresplot2 = fig3.add_subplot(1,2,1)
    scoresplot2.set_xlabel('Scores PC 2 (' + str(pca.explained_variance_ratio_[1]*100) + '%)')
    scoresplot2.set_ylabel('Scores PC 3 (' + str(pca.explained_variance_ratio_[2]*100) + '%)')
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
    loadingsplot2.plot(np.arange(X1.shape[1]), np.zeros(X1.shape[1]),'r--')
    loadingsplot2.set_xlim(0, X1.shape[1])
    loadingsplot2.legend()

def modelmetrics(model, X_train, y_train, X_test, y_test, cv=10):
    "output = modelmetrics(model, X_train, y_train, X_test, y_test, cv=10)"

    import pandas as pd
    import math
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import mean_squared_error, r2_score

    model.fit(X_train, y_train)
    y_train_predicted = model.predict(X_train)
    y_train_predicted_CV = cross_val_predict(model, X_train, y_train, cv=cv)
    y_test_predicted = model.predict(X_test)

    rmsec = math.sqrt(mean_squared_error(y_train, y_train_predicted))
    rmsecv = math.sqrt(mean_squared_error(y_train, y_train_predicted_CV))
    rmsep = math.sqrt(mean_squared_error(y_test, y_test_predicted))
    r2c = r2_score(y_train, y_train_predicted)
    r2cv = r2_score(y_train, y_train_predicted_CV)
    r2p = r2_score(y_test, y_test_predicted)

    output = pd.DataFrame({'Model':[model],
                           'RMSEC': [rmsec],
                           'R2C': [r2c],
                           'RMSECV': [rmsecv],
                           'R2CV': [r2cv],
                           'RMSEP': [rmsep],
                           'R2P': [r2p]})

    return output


def runPLS(X_train, y_train, X_test, y_test, n_components, cv=10, plot='off'):
    "output = runPLS(X_train, y_train, X_test, y_test, n_components, cv=10, plot='off')"

    import numpy as np
    from sklearn.cross_decomposition import PLSRegression

    model = PLSRegression(n_components=n_components)
    output = modelmetrics(model, X_train, y_train, X_test, y_test, cv=cv)

    if plot == 'on':
        import matplotlib.pyplot as plt
        from sklearn.model_selection import cross_val_predict
        y_train_predicted_CV = cross_val_predict(model, X_train, y_train, cv=cv)
        y_test_predicted = model.predict(X_test)
        plt.style.use('ggplot')
        fig = plt.figure(figsize = (6,6), dpi=300)
        mxp = fig.add_subplot(1,1,1)
        mxp.set_xlabel('Measured')
        mxp.set_ylabel('Predicted')
        mxp.set_xlim(min(min(np.array(y_train)),
                         min(y_train_predicted_CV),
                         min(np.array(y_test)),
                         min(y_test_predicted)),
                     max(max(np.array(y_train)),
                         max(y_train_predicted_CV),
                         max(np.array(y_test)),
                         max(y_test_predicted)))
        mxp.set_ylim(min(min(np.array(y_train)),
                         min(y_train_predicted_CV),
                         min(np.array(y_test)),
                         min(y_test_predicted)),
                     max(max(np.array(y_train)),
                         max(y_train_predicted_CV),
                         max(np.array(y_test)),
                         max(y_test_predicted)))
        mxp.scatter(y_train, y_train_predicted_CV, label = 'Train set')
        mxp.scatter(y_test, y_test_predicted, label = 'Test set')
        index1 = np.arange(len(y_train))
        index2 = np.arange(len(y_test))
        for xi, yi, indexi in zip(np.array(y_train), np.array(y_train_predicted_CV), index1):
            mxp.annotate(str(indexi), xy = (xi, yi))
        for xi, yi, indexi in zip(np.array(y_test), np.array(y_test_predicted), index2):
            mxp.annotate(str(indexi), xy = (xi, yi))
        mxp.legend()

    return output

def autoPLS(X_train, y_train, X_test, y_test, max_components=20, cv=10, plot='off'):
    "output = autoPLS(X_train, y_train, X_test, y_test, max_components=20, cv=10, plot='off')"

    import pandas as pd
    import numpy as np

    output = pd.DataFrame()

    for n_components in range(1, min(max_components+1, X_train.shape[1]+1)):
        output = output.append(runPLS(X_train, y_train, X_test, y_test,
                                      n_components=n_components, cv=cv, plot='off'),
                               ignore_index=True)
    diff = np.diff(output['RMSECV'])
    for n_components in range(1, min(max_components, X_train.shape[1])):
        if (diff/output['RMSECV'].iloc[len(diff)])[n_components-1] > -0.1: break

    if plot == 'on':
        import matplotlib.pyplot as plt
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_predict
        plt.style.use('ggplot')
        fig1 = plt.figure(figsize = (12,6), dpi=300)
        RMSEplot = fig1.add_subplot(1,2,1)
        RMSEplot.plot(np.arange(max_components)+1, output['RMSEC'], label = 'RMSEC')
        RMSEplot.plot(np.arange(max_components)+1, output['RMSECV'], label = 'RMSECV')
        RMSEplot.plot(np.arange(max_components)+1, output['RMSEP'], label = 'RMSEP')
        RMSEplot.set_xlabel('Components')
        RMSEplot.set_ylabel('A. U.')
        RMSEplot.set_xlim(1, max_components)
        RMSEplot.set_xticks(np.arange(max_components)+1)
        RMSEplot.legend()

        R2plot = fig1.add_subplot(1,2,2)
        R2plot.plot(np.arange(max_components)+1, output['R2C'], label = 'R2C')
        R2plot.plot(np.arange(max_components)+1, output['R2CV'], label = 'R2CV')
        R2plot.plot(np.arange(max_components)+1, output['R2P'], label = 'R2P')
        R2plot.set_xlabel('Components')
        R2plot.set_ylabel('A. U.')
        R2plot.set_xlim(1, max_components)
        R2plot.set_xticks(np.arange(max_components)+1)
        R2plot.legend()

        RMSEplot.plot(n_components, output['RMSEC'].iloc[n_components-1], 'P', ms=10, mfc='red')
        RMSEplot.plot(n_components, output['RMSECV'].iloc[n_components-1], 'P', ms=10, mfc='red')
        RMSEplot.plot(n_components, output['RMSEP'].iloc[n_components-1], 'P', ms=10, mfc='red')
        R2plot.plot(n_components, output['R2C'].iloc[n_components-1], 'P', ms=10, mfc='red')
        R2plot.plot(n_components, output['R2CV'].iloc[n_components-1], 'P', ms=10, mfc='red')
        R2plot.plot(n_components, output['R2P'].iloc[n_components-1], 'P', ms=10, mfc='red')

        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train, y_train)
        y_train_predicted_CV = cross_val_predict(pls, X_train, y_train, cv=cv)
        y_test_predicted = pls.predict(X_test)

        fig3 = plt.figure(figsize = (6,6), dpi=300)
        mxp = fig3.add_subplot(1,1,1)
        mxp.set_xlabel('Measured')
        mxp.set_ylabel('Predicted')
        mxp.set_xlim(min(min(np.array(y_train)),
                         min(y_train_predicted_CV),
                         min(np.array(y_test)),
                         min(y_test_predicted)),
                     max(max(np.array(y_train)),
                         max(y_train_predicted_CV),
                         max(np.array(y_test)),
                         max(y_test_predicted)))
        mxp.set_ylim(min(min(np.array(y_train)),
                         min(y_train_predicted_CV),
                         min(np.array(y_test)),
                         min(y_test_predicted)),
                     max(max(np.array(y_train)),
                         max(y_train_predicted_CV),
                         max(np.array(y_test)),
                         max(y_test_predicted)))
        mxp.scatter(y_train, y_train_predicted_CV, label = 'Train set')
        mxp.scatter(y_test, y_test_predicted, label = 'Test set')
        index1 = np.arange(len(y_train))
        index2 = np.arange(len(y_test))
        for xi, yi, indexi in zip(np.array(y_train), np.array(y_train_predicted_CV), index1):
            mxp.annotate(str(indexi), xy = (xi, yi))
        for xi, yi, indexi in zip(np.array(y_test), np.array(y_test_predicted), index2):
            mxp.annotate(str(indexi), xy = (xi, yi))
        mxp.legend()

    output = output.iloc[n_components-1]
    output.name = 'autoPLS'
    output = pd.DataFrame(output).T

    return output

def autosgPLS(X_train, y_train, X_test, y_test, max_components, cv=10, plot='off'):
    "output = autosgPLS(X_train, y_train, X_test, y_test, max_components=20, cv=10, plot='off')"

    import pandas as pd
    from scipy.signal import savgol_filter

    sg = pd.DataFrame()
    temp = pd.DataFrame()
    output = pd.DataFrame()

    for window_length in range(1, int(X_test.shape[1]*0.1), 2):
        for deriv in range(0, 2):
            for polyorder in range(deriv, min(2, window_length)):
                X_train_sg = savgol_filter(X_train, window_length, polyorder, deriv)
                X_test_sg = savgol_filter(X_test, window_length, polyorder, deriv)
                sg = sg.append((pd.DataFrame({'Window': [window_length],
                                              'PolyOrder': [polyorder],
                                              'Derivative': [deriv]})),
                               ignore_index=True)
                temp = autoPLS(X_train_sg, y_train, X_test_sg, y_test,
                                max_components=max_components,
                                cv=cv,
                                plot='off')
                output = output.append(temp, ignore_index=True)

    output = output.join(sg)

    output = output.sort_values(by=['RMSECV'])
    output = output.iloc[0]
    output.name = 'autosgPLS'
    output = pd.DataFrame(output).T

    if plot == 'on':
        X_train_sg = savgol_filter(X_train,
                                    int(output['Window']),
                                    int(output['PolyOrder']),
                                    int(output['Derivative']))
        X_test_sg = savgol_filter(X_test,
                                    int(output['Window']),
                                    int(output['PolyOrder']),
                                    int(output['Derivative']))
        autoPLS(X_train_sg, y_train, X_test_sg, y_test, max_components, cv=cv, plot='on')

    return output

def varsel(X_train, y_train, X_test, y_test, max_components, estimator='all', cv=10, plot='off'):
    "[output, varsel] = varsel(X_train, y_train, X_test, y_test, n_components, estimator='all', cv=10, plot='off')"

    import pandas as pd
    from sklearn.feature_selection import RFECV

    info = pd.DataFrame()
    varsel = pd.DataFrame()
    output = pd.DataFrame()

    if estimator == 'PLSRegression' or 'all':
        from sklearn.cross_decomposition import PLSRegression
        for n_components in range(1, max_components+1):
            model = PLSRegression(n_components=n_components)
            rfecv = RFECV(estimator=model, scoring='neg_root_mean_squared_error', cv=cv, step=int(X_train.shape[1]*0.05), min_features_to_select=n_components)
            rfecv.fit(X_train, y_train.values.ravel())
            info = info.append(pd.DataFrame({'InfoVec': [model], 'nVars': [rfecv.n_features_]}), ignore_index=True)
            varsel = varsel.append(pd.DataFrame(rfecv.support_).T, ignore_index=True)

    if estimator == 'RandomForestRegressor' or 'all':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(max_depth=X_train.shape[1]*0.5)
        model.fit(X_train, y_train.values.ravel())
        rfecv = RFECV(estimator=model, scoring='neg_root_mean_squared_error', cv=cv, step=int(X_train.shape[1]*0.1))
        rfecv.fit(X_train, y_train.values.ravel())
        info = info.append(pd.DataFrame({'InfoVec': [model], 'nVars': [rfecv.n_features_]}), ignore_index=True)
        varsel = varsel.append(pd.DataFrame(rfecv.support_).T, ignore_index=True)

    if estimator == 'Ridge' or 'all':
        from sklearn import linear_model
        model = linear_model.Ridge(alpha=.5)
        model.fit(X_train, y_train.values.ravel())
        rfecv = RFECV(estimator=model, scoring='neg_root_mean_squared_error', cv=cv, step=int(X_train.shape[1]*0.1))
        rfecv.fit(X_train, y_train.values.ravel())
        info = info.append(pd.DataFrame({'InfoVec': [model], 'nVars': [rfecv.n_features_]}), ignore_index=True)
        varsel = varsel.append(pd.DataFrame(rfecv.support_).T, ignore_index=True)

    if estimator == 'Lasso' or 'all':
        from sklearn import linear_model
        model = linear_model.Lasso(alpha=0.1)
        model.fit(X_train, y_train.values.ravel())
        rfecv = RFECV(estimator=model, scoring='neg_root_mean_squared_error', cv=cv, step=int(X_train.shape[1]*0.1))
        rfecv.fit(X_train, y_train.values.ravel())
        info = info.append(pd.DataFrame({'InfoVec': [model], 'nVars': [rfecv.n_features_]}), ignore_index=True)
        varsel = varsel.append(pd.DataFrame(rfecv.support_).T, ignore_index=True)

    if estimator == 'LassoLars' or 'all':
        from sklearn import linear_model
        model = linear_model.LassoLars(alpha=.1)
        model.fit(X_train, y_train.values.ravel())
        rfecv = RFECV(estimator=model, scoring='neg_root_mean_squared_error', cv=cv, step=int(X_train.shape[1]*0.1))
        rfecv.fit(X_train, y_train.values.ravel())
        info = info.append(pd.DataFrame({'InfoVec': [model], 'nVars': [rfecv.n_features_]}), ignore_index=True)
        varsel = varsel.append(pd.DataFrame(rfecv.support_).T, ignore_index=True)

    if estimator == 'BayesianRidge' or 'all':
        from sklearn import linear_model
        model = linear_model.BayesianRidge()
        model.fit(X_train, y_train.values.ravel())
        rfecv = RFECV(estimator=model, scoring='neg_root_mean_squared_error', cv=cv, step=int(X_train.shape[1]*0.1))
        rfecv.fit(X_train, y_train.values.ravel())
        info = info.append(pd.DataFrame({'InfoVec': [model], 'nVars': [rfecv.n_features_]}), ignore_index=True)
        varsel = varsel.append(pd.DataFrame(rfecv.support_).T, ignore_index=True)

    for temp in range(varsel.shape[0]):
        output = output.append(autoPLS(X_train.iloc[:,varsel.iloc[temp,:].values], y_train,
                                      X_test.iloc[:,varsel.iloc[temp,:].values], y_test,
                                      max_components=max_components, cv=10, plot='off'),
                               ignore_index=True)
    output = info.join(output).sort_values(by=['RMSECV'])
    varsel = varsel.iloc[output.index.values, :]

    if plot == 'on':
        import matplotlib.pyplot as plt
        import numpy as np
        fig = plt.figure(figsize = (6,6), dpi=300)
        select = fig.add_subplot(1,1,1) 
        select.plot(X_train.T)
        select.vlines(np.arange(0, varsel.shape[1])[varsel.iloc[0].values], ymin=min(X_train.min(0)), ymax=max(X_train.max(0)),linestyles='dotted', color='r')
        autoPLS(X_train.iloc[:,varsel.iloc[0,:].values], y_train, X_test.iloc[:,varsel.iloc[0,:].values], y_test, max_components=min(max_components, X_train.iloc[:,varsel.iloc[0,:].values].shape[1]), cv=cv, plot='on')

    return [output, varsel]
