"""Provide Gradient Boosting Regressor
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from optuna.distributions import IntUniformDistribution, \
    UniformDistribution, CategoricalDistribution, LogUniformDistribution

params = {
    'regressor__hidden_layer_sizes': CategoricalDistribution([
        (3,), (4,), (5,), (10,),
        (3,3), (4,4), (5,5), (10,10),
        (3,3,3), (4,4,4), (5,5,5), (10,10,3),
        ]),
    'regressor__alpha': LogUniformDistribution(1e-5, 1e-1),
}

model = TransformedTargetRegressor(
    regressor=MLPRegressor(max_iter=1000, solver='sgd', learning_rate_init=1e-2),
    func=np.log1p, inverse_func=np.expm1)
