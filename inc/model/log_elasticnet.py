"""Provide log elasticnet regression
"""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet
from optuna.distributions import \
    UniformDistribution, LogUniformDistribution, IntUniformDistribution
# import warnings
# warnings.filterwarnings('ignore') 

params = {
    'regressor__alpha': LogUniformDistribution(1e-6, 1e3),
    'regressor__l1_ratio': UniformDistribution(0.0, 1.0),
    # 'regressor__max_iter': IntUniformDistribution(500, 1500),
}

model = TransformedTargetRegressor(
    regressor=ElasticNet(tol=1e-1, max_iter=1e4), func=np.log1p, inverse_func=np.expm1)
