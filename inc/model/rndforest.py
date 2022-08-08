"""Provide Random Forest
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from optuna.distributions import \
    CategoricalDistribution, IntUniformDistribution, UniformDistribution

params = {
    'n_estimators': IntUniformDistribution(100, 1000),
    'max_depth': IntUniformDistribution(10, 1000),
    'min_samples_split': IntUniformDistribution(3, 300),
    'min_samples_leaf': IntUniformDistribution(3, 300),
    'max_features': UniformDistribution(0.1, 1.0),
    'max_leaf_nodes': IntUniformDistribution(5, 1000),
    'max_samples': UniformDistribution(0.1, 1.0),
}

model = RandomForestRegressor()
