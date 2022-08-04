"""Provide log transformed linear regression
"""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LassoLars
from optuna.distributions import \
    UniformDistribution, LogUniformDistribution, IntUniformDistribution
import warnings
warnings.filterwarnings('ignore') 

params = {
    'regressor__alpha': UniformDistribution(0.1, 2.0),
    'regressor__eps': LogUniformDistribution(1e-18, 1e-14),
    'regressor__max_iter': IntUniformDistribution(200, 1500),
}

model = TransformedTargetRegressor(
    regressor=LassoLars(), func=np.log1p, inverse_func=np.expm1)
