#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:32:37 2021

@author: caliariitalo
"""
from chemometrics import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from statistics import variance
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.cross_decomposition import PLSRegression

X = (np.array(X)).squeeze()
y = (np.array(y)).squeeze()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

[output0, vs0] = varsel(X_train, y_train, X_test, y_test, info='all', step=0.05, max_components=10, cv=10, plot='on')

plt.plot(vector)



[output1, vs1] = varsel(X_train, y_train, X_test, y_test, info='PLSReg', step=0.05, max_components=10, cv=10, plot='on')
[output2, vs2] = varsel(X_train, y_train, X_test, y_test, info='Loadings', step=0.05, max_components=10, cv=10, plot='on')
[output3, vs3] = varsel(X_train, y_train, X_test, y_test, info='Weigths', step=0.05, max_components=10, cv=10, plot='on')
[output4, vs4] = varsel(X_train, y_train, X_test, y_test, info='std', step=0.05, max_components=10, cv=10, plot='on')
[output5, vs5] = varsel(X_train, y_train, X_test, y_test, info='SQR', step=0.05, max_components=10, cv=10, plot='on')
[output6, vs6] = varsel(X_train, y_train, X_test, y_test, info='LinReg', step=0.05, max_components=10, cv=10, plot='on')
[output7, vs7] = varsel(X_train, y_train, X_test, y_test, info='all', step=0.05, max_components=10, cv=10, plot='on')
[output8, vs8] = varsel(X_train, y_train, X_test, y_test, info='all', step=0.05, max_components=10, cv=10, plot='on')



print('A base de dados apresenta {} registros (imóveis) e {} variáveis'.format(dados.shape[0], dados.shape[1]))

index = ['Linha' + str(i) for i in range(5)]
columns = ['Coluna' + str(i) for i in range(3)]
df2 = pd.DataFrame(data = data, index = index, columns = columns)

residencial = ['Quitinete', 
                'Casa',
                'Apartamento',
                'Casa de Condomínio',
                'Casa de Vila']

selecao = dados['Tipo'].isin(residencial)

dados_residencial = dados[selecao]

list('321')

df = pd.DataFrame(data, list('321'), list('ZYX'))

df.sort_values(by = ['X','Y'], inplace = True)

#Selecione somente os imóveis classificados com tipo 'Apartamento'.
selecao = dados ['Tipo'] == 'Apartamento'
selecao

data = [(1, 2, 3, 4),
        (5, 6, 7, 8),
        (8, 10, 11, 12),
        (13, 14, 15, 16)]
df = pd.DataFrame(data, 'l1 l2 l3 l4'.split(), 'c1 c2 c3 c4'.split())

dados.fillna(0, inplace = True)

dados = dados.fillna({'Condominio': 0, 'IPTU': 0})

sexo = alunos.groupby('Sexo')

grupo_bairro['Valor'].describe().round(2)

dados.boxplot(['Valor'])


Q1 = valor.quantile(.25)
Q3 = valor.quantile(.75)
IIQ = Q3 - Q1
limite_inferior = Q1 - 1.5 * IIQ
limite_superior = Q3 + 1.5 * IIQ


dados.hist(['Valor'])
dados_new.hist(['Valor'])



dados_new = pd.DataFrame()
for tipo in grupo_tipo.groups.keys():
    eh_tipo = dados['Tipo'] == tipo
    eh_dentro_limite = (dados['Valor'] >= limite_inferior[tipo]) & (dados['Valor'] <= limite_superior[tipo])
    selecao = eh_tipo & eh_dentro_limite
    dados_selecao = dados[selecao]
    dados_new = pd.concat([dados_new, dados_selecao])
    
dist_freq_qualitativas.rename(index = {0: 'Masculino', 1: 'Feminino'}, inplace = True)   
dist_freq_qualitativas.rename(index = {0: 'Masculino', 1: 'Feminino'}, inplace = True)
dist_freq_qualitativas.rename_axis('Sexo', axis = 'columns', inplace = True)
    
    
    