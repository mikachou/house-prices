"""Provide log transformed decision tree regression
"""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor
from optuna.distributions import IntLogUniformDistribution

params = {
    'regressor__max_depth': IntLogUniformDistribution(5, 200),
    'regressor__min_samples_split': IntLogUniformDistribution(5, 200),
    'regressor__min_samples_leaf': IntLogUniformDistribution(5, 200),
}

model = TransformedTargetRegressor(
        regressor=DecisionTreeRegressor(), func=np.log1p, inverse_func=np.expm1)
