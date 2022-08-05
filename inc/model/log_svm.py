"""Provide log svm regression
"""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR
from optuna.distributions import \
    UniformDistribution, LogUniformDistribution, IntUniformDistribution
# import warnings
# warnings.filterwarnings('ignore') 

params = {
    'regressor__gamma': UniformDistribution(1e-6, 1e3),
    'regressor__C': UniformDistribution(1e-6, 1e3),
    'regressor__epsilon': UniformDistribution(1e-6, 1e3),
}

model = TransformedTargetRegressor(
    regressor=SVR(), func=np.log1p, inverse_func=np.expm1)
