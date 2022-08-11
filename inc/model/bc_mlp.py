"""Provide Gradient Boosting Regressor
"""
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from optuna.distributions import \
    CategoricalDistribution, LogUniformDistribution

train = pd.read_csv('./data/train.csv')

tr = PowerTransformer(method='box-cox')
tr.fit(train['SalePrice'].values.reshape(-1, 1))

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
    func=tr.transform, inverse_func=tr.inverse_transform)
