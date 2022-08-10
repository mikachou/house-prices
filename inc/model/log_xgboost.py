"""Provide log xgboost regression
"""
import numpy as np
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from optuna.distributions import \
    CategoricalDistribution, IntUniformDistribution, \
    UniformDistribution, LogUniformDistribution

params = {
    # 'regressor__n_estimators': IntUniformDistribution(50, 500),
    'regressor__max_depth': IntUniformDistribution(3, 1000),
    'regressor__max_leaves': IntUniformDistribution(3, 1500),
    'regressor__learning_rate': LogUniformDistribution(1e-4, 1e-1),
    'regressor__subsample': UniformDistribution(0.1, 1.0),
    'regressor__colsample_bytree': UniformDistribution(0.1, 1.0),
    'regressor__colsample_bylevel': UniformDistribution(0.1, 1.0),
    'regressor__colsample_bynode': UniformDistribution(0.1, 1.0),
}

model = TransformedTargetRegressor(
    regressor=XGBRegressor(n_estimators=1000),
    func=np.log1p, inverse_func=np.expm1)
