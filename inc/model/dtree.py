"""Provide decision tree regression
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from optuna.distributions import IntLogUniformDistribution

params = {
    'max_depth': IntLogUniformDistribution(5, 200),
    'min_samples_split': IntLogUniformDistribution(5, 200),
    'min_samples_leaf': IntLogUniformDistribution(5, 200),
}

model = DecisionTreeRegressor()
