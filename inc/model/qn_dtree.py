"""Provide quantile-normal transformed decision tree regression
"""
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor
from optuna.distributions import IntLogUniformDistribution

train = pd.read_csv('./data/train.csv')

tr = QuantileTransformer(output_distribution='normal')
tr.fit(train['SalePrice'].values.reshape(-1, 1))

params = {
    'regressor__max_depth': IntLogUniformDistribution(5, 200),
    'regressor__min_samples_split': IntLogUniformDistribution(5, 200),
    'regressor__min_samples_leaf': IntLogUniformDistribution(5, 200),
}

model = TransformedTargetRegressor(
    regressor=DecisionTreeRegressor(),
    func=tr.transform, inverse_func=tr.inverse_transform)
