import pandas as pd
import numpy as np
import scipy.stats as stats

A = (21.8, 21.9, 21.7, 21.6, 21.7, 21.5, 47)
B = (21.7, 21.4, 21.5, 21.5)
C = (21.9, 21.8, 21.8, 21.6, 21.5)
D = (21.9, 21.7, 21.8, 21.7, 21.6, 21.8)

fvalue, pvalue = stats.f_oneway(A, B, C, D)