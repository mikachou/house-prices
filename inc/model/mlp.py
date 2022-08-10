"""Provide Gradient Boosting Regressor
"""
from sklearn.neural_network import MLPRegressor
from optuna.distributions import CategoricalDistribution, LogUniformDistribution

params = {
    'hidden_layer_sizes': CategoricalDistribution([
        (3,), (4,), (5,), (10,),
        (3,3), (4,4), (5,5), (10,10),
        (3,3,3), (4,4,4), (5,5,5), (10,10,3),
        ]),
    'alpha': LogUniformDistribution(1e-5, 1e-1),
}

model = MLPRegressor(max_iter=1000, solver='sgd', learning_rate_init=1e-2)
