"""Provide box-cox lasso regression
"""
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Lasso
from optuna.distributions import \
    UniformDistribution, LogUniformDistribution, IntUniformDistribution
# import warnings
# warnings.filterwarnings('ignore') 

train = pd.read_csv('./data/train.csv')

tr = PowerTransformer(method='box-cox')
tr.fit(train['SalePrice'].values.reshape(-1, 1))

params = {
    'regressor__alpha': LogUniformDistribution(1e-6, 1e3),
}

model = TransformedTargetRegressor(
    regressor=Lasso(tol=1e-1, max_iter=1e4),
    func=tr.transform, inverse_func=tr.inverse_transform)
