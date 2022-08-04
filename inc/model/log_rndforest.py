"""Provide log Random Forest
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from optuna.distributions import \
    CategoricalDistribution, IntUniformDistribution, UniformDistribution

params = {
    'regressor__n_estimators': IntUniformDistribution(100, 500),
    'regressor__max_depth': IntUniformDistribution(10, 100),
    'regressor__min_samples_split': IntUniformDistribution(3, 300),
    'regressor__min_samples_leaf': IntUniformDistribution(3, 300),
    'regressor__max_features': UniformDistribution(0.1, 1.0),
    'regressor__max_leaf_nodes': IntUniformDistribution(5, 300),
    'regressor__max_samples': UniformDistribution(0.1, 1.0),
}

model = TransformedTargetRegressor(
        regressor=RandomForestRegressor(), func=np.log1p, inverse_func=np.expm1)
