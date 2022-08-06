"""Provide yeo-johnson xgboost regression
"""
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
from optuna.distributions import \
    CategoricalDistribution, IntUniformDistribution, \
    UniformDistribution, LogUniformDistribution

train = pd.read_csv('./data/train.csv')

tr = PowerTransformer(method='yeo-johnson')
tr.fit(train['SalePrice'].values.reshape(-1, 1))

params = {
    'regressor__n_estimators': IntUniformDistribution(50, 500),
    'regressor__max_depth': IntUniformDistribution(3, 300),
    'regressor__max_leaves': IntUniformDistribution(3, 1500),
    'regressor__learning_rate': LogUniformDistribution(1e-4, 1e-1),
    'regressor__subsample': UniformDistribution(0.1, 1.0),
    'regressor__colsample_bytree': UniformDistribution(0.1, 1.0),
    'regressor__colsample_bylevel': UniformDistribution(0.1, 1.0),
    'regressor__colsample_bynode': UniformDistribution(0.1, 1.0),
}

model = TransformedTargetRegressor(
    regressor=XGBRegressor(),
    func=tr.transform, inverse_func=tr.inverse_transform)
