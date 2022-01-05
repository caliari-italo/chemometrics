# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 12:07:54 2022

@author: calia
"""

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from chemometrics import *
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

X_train = X_train.iloc[:,::3]
X_test = X_test.iloc[:,::3]
















# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
  
# Load Real State Data
data = pd.read_csv('/content/Real estate.csv')
data.head()
  
# Fit a OLS regression variable
model = smf.ols(formula=' Y ~ X3 + X2', data= data )
results = model.fit()
print(results.summary())
  
# Get different Variables for diagnostic
residuals = results.resid
fitted_value = results.fittedvalues
stand_resids = results.resid_pearson
influence = results.get_influence()
leverage = influence.hat_matrix_diag
  
# PLot different diagnostic plots
plt.rcParams["figure.figsize"] = (20,15)
fig, ax = plt.subplots(nrows=2, ncols=2)
  
plt.style.use('seaborn')
  
# Residual vs Fitted Plot
sns.scatterplot(x=fitted_value, y=residuals, ax=ax[0, 0])
ax[0, 0].axhline(y=0, color='grey', linestyle='dashed')
ax[0, 0].set_xlabel('Fitted Values')
ax[0, 0].set_ylabel('Residuals')
ax[0, 0].set_title('Residuals vs Fitted Fitted')
  
# Normal Q-Q plot
sm.qqplot(residuals, fit=True, line='45',ax=ax[0, 1], c='#4C72B0')
ax[0, 1].set_title('Normal Q-Q')
  
# Scale-Location Plot
sns.scatterplot(x=fitted_value, y=residuals, ax=ax[1, 0])
ax[1, 0].axhline(y=0, color='grey', linestyle='dashed')
ax[1, 0].set_xlabel('Fitted values')
ax[1, 0].set_ylabel('Sqrt(standardized residuals)')
ax[1, 0].set_title('Scale-Location Plot')
  
# Residual vs Leverage Plot
sns.scatterplot(x=leverage, y=stand_resids, ax=ax[1, 1])
ax[1, 1].axhline(y=0, color='grey', linestyle='dashed')
ax[1, 1].set_xlabel('Leverage')
ax[1, 1].set_ylabel('Sqrt(standardized residuals)')
ax[1, 1].set_title('Residuals vs Leverage Plot')
  
  
plt.tight_layout()
plt.show()
  
# PLot Cook's distance plot
sm.graphics.influence_plot(results, criterion="cooks")