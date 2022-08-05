"""Provide Gradient Boosting Regressor
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from optuna.distributions import IntUniformDistribution, UniformDistribution

params = {
    'regressor__learning_rate': UniformDistribution(1e-3, 1.0),
    'regressor__n_estimators': IntUniformDistribution(50, 500),
    'regressor__subsample': UniformDistribution(0.1, 1.0),
    'regressor__max_depth': IntUniformDistribution(3, 300),
    'regressor__min_samples_split': IntUniformDistribution(3, 300),
    'regressor__min_samples_leaf': IntUniformDistribution(3, 300),
    'regressor__max_features': UniformDistribution(0.1, 1.0),
    'regressor__max_leaf_nodes': IntUniformDistribution(3, 300),
}

model = TransformedTargetRegressor(
    regressor=GradientBoostingRegressor(), func=np.log1p, inverse_func=np.expm1)
