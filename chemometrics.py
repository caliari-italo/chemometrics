#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usefull Chemometrics routines
@author: caliariitalo
"""

def runPCA(X):
    "runPCA(X)"

    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    X = (np.array(X)).squeeze()
    
    index = np.arange(X.shape[0])
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

    euclidean = np.zeros(X.shape[0])
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

    X1 = (np.array(X1)).squeeze()
    X2 = (np.array(X2)).squeeze()

    index1 = np.arange(X1.shape[0])
    index2 = np.arange(X2.shape[0])

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

    import numpy as np
    import pandas as pd
    import math
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import mean_squared_error, r2_score

    X_train = (np.array(X_train)).squeeze()
    y_train = (np.array(y_train)).squeeze()
    X_test = (np.array(X_test)).squeeze()
    y_train = (np.array(y_train)).squeeze()

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

    X_train = (np.array(X_train)).squeeze()
    y_train = (np.array(y_train)).squeeze()
    X_test = (np.array(X_test)).squeeze()
    y_train = (np.array(y_train)).squeeze()

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
        mxp.set_xlim(min(min(y_train),
                         min(y_train_predicted_CV),
                         min(y_test),
                         min(y_test_predicted)),
                     max(max(y_train),
                         max(y_train_predicted_CV),
                         max(y_test),
                         max(y_test_predicted)))
        mxp.set_ylim(min(min(y_train),
                         min(y_train_predicted_CV),
                         min(y_test),
                         min(y_test_predicted)),
                     max(max(y_train),
                         max(y_train_predicted_CV),
                         max(y_test),
                         max(y_test_predicted)))
        mxp.scatter(y_train, y_train_predicted_CV, label = 'Train set')
        mxp.scatter(y_test, y_test_predicted, label = 'Test set')
        index1 = np.arange(len(y_train))
        index2 = np.arange(len(y_test))
        for xi, yi, indexi in zip(y_train, y_train_predicted_CV, index1):
            mxp.annotate(str(indexi), xy = (xi, yi))
        for xi, yi, indexi in zip(y_test, y_test_predicted, index2):
            mxp.annotate(str(indexi), xy = (xi, yi))
        mxp.legend()

    return output

def autoPLS(X_train, y_train, X_test, y_test, max_components=20, cv=10, plot='off'):
    "output = autoPLS(X_train, y_train, X_test, y_test, max_components=20, cv=10, plot='off')"

    import numpy as np
    import pandas as pd

    X_train = (np.array(X_train)).squeeze()
    y_train = (np.array(y_train)).squeeze()
    X_test = (np.array(X_test)).squeeze()
    y_train = (np.array(y_train)).squeeze()

    output = pd.DataFrame()

    for n_components in range(1, min(max_components+1, X_train.shape[1]+1)):
        output = output.append(runPLS(X_train, y_train, X_test, y_test,
                                      n_components=n_components, cv=cv, plot='off'),
                               ignore_index=True)
    diff = np.diff(output['RMSECV'])
    for n_components in range(1, min(max_components, X_train.shape[1])):
        if (diff/output['RMSECV'].iloc[len(diff)])[n_components-1] > -0.25: break

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
        mxp.set_xlim(min(min(y_train),
                         min(y_train_predicted_CV),
                         min(y_test),
                         min(y_test_predicted)),
                     max(max(y_train),
                         max(y_train_predicted_CV),
                         max(y_test),
                         max(y_test_predicted)))
        mxp.set_ylim(min(min(y_train),
                         min(y_train_predicted_CV),
                         min(y_test),
                         min(y_test_predicted)),
                     max(max(y_train),
                         max(y_train_predicted_CV),
                         max(y_test),
                         max(y_test_predicted)))
        mxp.scatter(y_train, y_train_predicted_CV, label = 'Train set')
        mxp.scatter(y_test, y_test_predicted, label = 'Test set')
        index1 = np.arange(len(y_train))
        index2 = np.arange(len(y_test))
        for xi, yi, indexi in zip(y_train, y_train_predicted_CV, index1):
            mxp.annotate(str(indexi), xy = (xi, yi))
        for xi, yi, indexi in zip(y_test, y_test_predicted, index2):
            mxp.annotate(str(indexi), xy = (xi, yi))
        mxp.legend()

    output = output.iloc[n_components-1]
    output.name = 'autoPLS'
    output = pd.DataFrame(output).T

    return output

def autosgPLS(X_train, y_train, X_test, y_test, max_components, cv=10, plot='off'):
    "output = autosgPLS(X_train, y_train, X_test, y_test, max_components=20, cv=10, plot='off')"

    import numpy as np
    import pandas as pd
    from scipy.signal import savgol_filter

    X_train = (np.array(X_train)).squeeze()
    y_train = (np.array(y_train)).squeeze()
    X_test = (np.array(X_test)).squeeze()
    y_train = (np.array(y_train)).squeeze()
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

def varsel(X_train, y_train, X_test, y_test, info='all', step=0.05, max_components=10, cv=10, plot='off'):
    "[output, vs] = varsel(X_train, y_train, X_test, y_test, info='all', step=0.05, max_components=10, cv=10, plot='off')"

    import numpy as np
    import pandas as pd

    output = pd.DataFrame()
    vs = pd.DataFrame()

    if info == 'PLSReg' or info == 'all':
        from sklearn.cross_decomposition import PLSRegression
        for n_components in range(1, max_components+1):
            model = PLSRegression(n_components=n_components)
            model.fit(X_train, y_train)
            vector = abs(model.coef_).squeeze()
            vector = np.flip(vector.argsort(0))
            temp_output = pd.DataFrame()
            temp_vs = pd.DataFrame()
            for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
                var = vector[0:window]
                temp = pd.concat([pd.DataFrame({'InfoVector': ['PLSReg n_components=' + str(n_components)], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
                temp_output = temp_output.append(temp, ignore_index=True)
                temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
            output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
            vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'Loadings' or info == 'all':
        from sklearn.cross_decomposition import PLSRegression
        model = PLSRegression(n_components=max_components)
        model.fit(X_train, y_train)
        for n_components in range(1, max_components+1):
            vector = abs(model.x_loadings_)[:,:n_components].sum(1).squeeze()
            vector = np.flip(vector.argsort(0))
            temp_output = pd.DataFrame()
            temp_vs = pd.DataFrame()
            for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
                var = vector[0:window]
                temp = pd.concat([pd.DataFrame({'InfoVector': ['Loadings n_components=' + str(n_components)], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
                temp_output = temp_output.append(temp, ignore_index=True)
                temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
            output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
            vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'Weigths' or info == 'all':
        from sklearn.cross_decomposition import PLSRegression
        model = PLSRegression(n_components=max_components)
        model.fit(X_train, y_train)
        for n_components in range(1, max_components+1):
            vector = abs(model.x_weights_)[:,:n_components].sum(1).squeeze()
            vector = np.flip(vector.argsort(0))
            temp_output = pd.DataFrame()
            temp_vs = pd.DataFrame()
            for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
                var = vector[0:window]
                temp = pd.concat([pd.DataFrame({'InfoVector': ['Weigths n_components=' + str(n_components)], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
                temp_output = temp_output.append(temp, ignore_index=True)
                temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
            output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
            vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'XVar' or info == 'all':
        from sklearn.cross_decomposition import PLSRegression
        vector = X_train.var(0)
        vector = np.flip(vector.argsort(0))
        temp_output = pd.DataFrame()
        temp_vs = pd.DataFrame()
        for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
            var = vector[0:window]
            temp = pd.concat([pd.DataFrame({'InfoVector': ['XVar'], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
            temp_output = temp_output.append(temp, ignore_index=True)
            temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
        output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
        vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'SQR' or info == 'all':
        from sklearn.cross_decomposition import PLSRegression
        for n_components in range(1, max_components+1):
            model = PLSRegression(n_components=n_components)
            model.fit(X_train, y_train)
            res = X_train - model.inverse_transform(model.x_scores_)
            vector = (res*res).sum(0).squeeze()
            vector = vector.argsort(0)
            temp_output = pd.DataFrame()
            temp_vs = pd.DataFrame()
            for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
                var = vector[0:window]
                temp = pd.concat([pd.DataFrame({'InfoVector': ['SQR n_components=' + str(n_components)], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
                temp_output = temp_output.append(temp, ignore_index=True)
                temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
            output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
            vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'URXy' or info == 'all':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        vector = np.zeros(X_train.shape[1])
        for i in range(X_train.shape[1]):
            model.fit(X_train[:, i].reshape((-1, 1)), y_train)
            vector[i] = model.coef_
        vector = abs(vector).squeeze()
        vector = np.flip(vector.argsort(0))
        temp_output = pd.DataFrame()
        temp_vs = pd.DataFrame()
        for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
            var = vector[0:window]
            temp = pd.concat([pd.DataFrame({'InfoVector': ['URXy'], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
            temp_output = temp_output.append(temp, ignore_index=True)
            temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
        output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
        vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])
        
    if info == 'Ridge' or info == 'all':
        from sklearn.linear_model import RidgeCV
        model = RidgeCV()
        model.fit(X_train, y_train)
        vector = abs(model.coef_).squeeze()
        vector = np.flip(vector.argsort(0))
        temp_output = pd.DataFrame()
        temp_vs = pd.DataFrame()
        for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
            var = vector[0:window]
            temp = pd.concat([pd.DataFrame({'InfoVector': ['Ridge'], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
            temp_output = temp_output.append(temp, ignore_index=True)
            temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
        output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
        vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'Lasso' or info == 'all': #Rever
        from sklearn.linear_model import LassoCV
        model = LassoCV()
        model.fit(X_train, y_train)
        vector = abs(model.coef_).squeeze()
        vector = np.flip(vector.argsort(0))
        temp_output = pd.DataFrame()
        temp_vs = pd.DataFrame()
        for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
            var = vector[0:window]
            temp = pd.concat([pd.DataFrame({'InfoVector': ['Lasso'], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
            temp_output = temp_output.append(temp, ignore_index=True)
            temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
        output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
        vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'ElasticNetCV' or info == 'all': #Warning
        from sklearn.linear_model import ElasticNetCV
        model = ElasticNetCV()
        model.fit(X_train, y_train)
        vector = abs(model.coef_).squeeze()
        vector = np.flip(vector.argsort(0))
        temp_output = pd.DataFrame()
        temp_vs = pd.DataFrame()
        for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
            var = vector[0:window]
            temp = pd.concat([pd.DataFrame({'InfoVector': ['ElasticNetCV'], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
            temp_output = temp_output.append(temp, ignore_index=True)
            temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
        output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
        vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'LassoLars' or info == 'all':
        from sklearn.linear_model import LassoLars
        model = LassoLars()
        model.fit(X_train, y_train)
        vector = abs(model.coef_).squeeze()
        vector = np.flip(vector.argsort(0))
        temp_output = pd.DataFrame()
        temp_vs = pd.DataFrame()
        for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
            var = vector[0:window]
            temp = pd.concat([pd.DataFrame({'InfoVector': ['LassoLars'], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
            temp_output = temp_output.append(temp, ignore_index=True)
            temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
        output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
        vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'OrthogonalMatchingPursuit' or info == 'all':
        from sklearn.linear_model import OrthogonalMatchingPursuit
        model = OrthogonalMatchingPursuit()
        model.fit(X_train, y_train)
        vector = abs(model.coef_).squeeze()
        vector = np.flip(vector.argsort(0))
        temp_output = pd.DataFrame()
        temp_vs = pd.DataFrame()
        for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
            var = vector[0:window]
            temp = pd.concat([pd.DataFrame({'InfoVector': ['OrthogonalMatchingPursuit'], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
            temp_output = temp_output.append(temp, ignore_index=True)
            temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
        output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
        vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'BayesianRidge' or info == 'all':
        from sklearn.linear_model import BayesianRidge
        model = BayesianRidge()
        model.fit(X_train, y_train)
        vector = abs(model.coef_).squeeze()
        vector = np.flip(vector.argsort(0))
        temp_output = pd.DataFrame()
        temp_vs = pd.DataFrame()
        for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
            var = vector[0:window]
            temp = pd.concat([pd.DataFrame({'InfoVector': ['BayesianRidge'], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
            temp_output = temp_output.append(temp, ignore_index=True)
            temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
        output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
        vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'ARDRegression' or info == 'all':
        from sklearn.linear_model import ARDRegression
        model = ARDRegression()
        model.fit(X_train, y_train)
        vector = abs(model.coef_).squeeze()
        vector = np.flip(vector.argsort(0))
        temp_output = pd.DataFrame()
        temp_vs = pd.DataFrame()
        for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
            var = vector[0:window]
            temp = pd.concat([pd.DataFrame({'InfoVector': ['ARDRegression'], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
            temp_output = temp_output.append(temp, ignore_index=True)
            temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
        output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
        vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'SGDRegressor' or info == 'all': #Rever
        from sklearn.linear_model import SGDRegressor
        model = SGDRegressor()
        model.fit(X_train, y_train)
        vector = abs(model.coef_).squeeze()
        vector = np.flip(vector.argsort(0))
        temp_output = pd.DataFrame()
        temp_vs = pd.DataFrame()
        for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
            var = vector[0:window]
            temp = pd.concat([pd.DataFrame({'InfoVector': ['SGDRegressor'], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
            temp_output = temp_output.append(temp, ignore_index=True)
            temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
        output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
        vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'PassiveAggressiveRegressor' or info == 'all':
        from sklearn.linear_model import PassiveAggressiveRegressor
        model = PassiveAggressiveRegressor()
        model.fit(X_train, y_train)
        vector = abs(model.coef_).squeeze()
        vector = np.flip(vector.argsort(0))
        temp_output = pd.DataFrame()
        temp_vs = pd.DataFrame()
        for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
            var = vector[0:window]
            temp = pd.concat([pd.DataFrame({'InfoVector': ['PassiveAggressiveRegressor'], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
            temp_output = temp_output.append(temp, ignore_index=True)
            temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
        output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
        vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'HuberRegressor' or info == 'all': #Warning
        from sklearn.linear_model import HuberRegressor
        model = HuberRegressor()
        model.fit(X_train, y_train)
        vector = abs(model.coef_).squeeze()
        vector = np.flip(vector.argsort(0))
        temp_output = pd.DataFrame()
        temp_vs = pd.DataFrame()
        for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
            var = vector[0:window]
            temp = pd.concat([pd.DataFrame({'InfoVector': ['HuberRegressor'], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
            temp_output = temp_output.append(temp, ignore_index=True)
            temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
        output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
        vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'LinearSVR' or info == 'all': #Rever
        from sklearn.svm import LinearSVR
        model = LinearSVR()
        model.fit(X_train, y_train)
        vector = abs(model.coef_).squeeze()
        vector = np.flip(vector.argsort(0))
        temp_output = pd.DataFrame()
        temp_vs = pd.DataFrame()
        for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
            var = vector[0:window]
            temp = pd.concat([pd.DataFrame({'InfoVector': ['LinearSVR'], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
            temp_output = temp_output.append(temp, ignore_index=True)
            temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
        output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
        vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if info == 'RandomForestRegressor' or info == 'all':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(max_depth=X_train.shape[1]*0.5)
        model.fit(X_train, y_train)
        vector = abs(model.feature_importances_).squeeze()
        vector = np.flip(vector.argsort(0))
        temp_output = pd.DataFrame()
        temp_vs = pd.DataFrame()
        for window in range(max(max_components, int(len(vector)*step)), len(vector)+1, int(len(vector)*step)):
            var = vector[0:window]
            temp = pd.concat([pd.DataFrame({'InfoVector': ['RandomForestRegressor'], 'nVars': [len(var)]}).reset_index(drop=True), (autoPLS(X_train[:, var], y_train, X_test[:, var], y_test, max_components=max_components, cv=cv, plot='off')).reset_index(drop=True)], axis=1)
            temp_output = temp_output.append(temp, ignore_index=True)
            temp_vs = temp_vs.append(pd.DataFrame(var).T, ignore_index=True)
        output = output.append(temp_output.iloc[np.argmin(temp_output['RMSECV'])])
        vs = vs.append(temp_vs.iloc[np.argmin(temp_output['RMSECV'])])

    if plot == 'on':
        best_vs = np.array(vs.iloc[np.argmin(output['RMSECV'])].dropna(), dtype='int')
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize = (6,6), dpi=300)
        select = fig.add_subplot(1,1,1) 
        select.plot(X_train.T)
        select.vlines(best_vs, ymin=min(X_train.min(0)), ymax=max(X_train.max(0)),linestyles='dotted', color='r')
        autoPLS(X_train[:,best_vs], y_train, X_test[:,best_vs], y_test, max_components=min(max_components, best_vs.shape[0]), cv=cv, plot='on')

    return [output, vs]
