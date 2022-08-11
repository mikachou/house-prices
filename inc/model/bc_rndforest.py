"""Provide box-cox Random Forest
"""
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from optuna.distributions import \
    IntUniformDistribution, UniformDistribution

train = pd.read_csv('./data/train.csv')

tr = PowerTransformer(method='box-cox')
tr.fit(train['SalePrice'].values.reshape(-1, 1))

params = {
    'regressor__n_estimators': IntUniformDistribution(100, 1000),
    'regressor__max_depth': IntUniformDistribution(10, 1000),
    'regressor__min_samples_split': IntUniformDistribution(3, 300),
    'regressor__min_samples_leaf': IntUniformDistribution(3, 300),
    'regressor__max_features': UniformDistribution(0.1, 1.0),
    'regressor__max_leaf_nodes': IntUniformDistribution(5, 1000),
    'regressor__max_samples': UniformDistribution(0.1, 1.0),
}

model = TransformedTargetRegressor(
    regressor=RandomForestRegressor(),
    func=tr.transform, inverse_func=tr.inverse_transform)
