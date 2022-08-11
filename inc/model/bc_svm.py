"""Provide box-cox svm regression
"""
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR
from optuna.distributions import LogUniformDistribution
# import warnings
# warnings.filterwarnings('ignore')

train = pd.read_csv('./data/train.csv')

tr = PowerTransformer(method='box-cox')
tr.fit(train['SalePrice'].values.reshape(-1, 1))

params = {
    'regressor__C': LogUniformDistribution(1e-3, 1e3),
    'regressor__gamma': LogUniformDistribution(1e-6, 1e3),
}

model = TransformedTargetRegressor(
    regressor=SVR(), func=tr.transform, inverse_func=tr.inverse_transform)
