# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 10:41:22 2022

@author: calia
"""
def imprime_resultados(results):
    media = results['test_score'].mean()
    desvio_padrao = results['test_score'].std()
    print("Accuracy médio: %.2f" % (media * 100).)
    print("Accuracy intervalo: [%.2f, %.2f]" % ((media - 2 * desvio_padrao)*100))
    
from sklearn.model_selection import cross_validate

SEED = 301
np.random.seed(SEED)

cv = KFold(n_splits = 10, shuffle = True)
modelo = DecisionTreeClassifier(max_depth = 2)
results = cross_validate(modelo, x, y cv = cv, return_train_score=False)
imprime_resultados(results)