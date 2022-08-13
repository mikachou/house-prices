"""Provide log svm regression
"""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR
from optuna.distributions import LogUniformDistribution, UniformDistribution

params = {
    'regressor__C': LogUniformDistribution(1e-3, 1e3),
    'regressor__gamma': LogUniformDistribution(1e-6, 1e3),
    'regressor__epsilon': LogUniformDistribution(1e-3, 1e0),
}

model = TransformedTargetRegressor(
    regressor=SVR(), func=np.log1p, inverse_func=np.expm1)
