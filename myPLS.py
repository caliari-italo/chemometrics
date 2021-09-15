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