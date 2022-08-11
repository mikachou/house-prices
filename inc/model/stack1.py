"""Stacking Regressor model
"""
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingRegressor
from optuna.distributions import \
    IntUniformDistribution, UniformDistribution, LogUniformDistribution

train = pd.read_csv('./data/train.csv')

tr = PowerTransformer(method='box-cox')
tr.fit(train['SalePrice'].values.reshape(-1, 1))

bc_catboost = TransformedTargetRegressor(
    regressor=CatBoostRegressor(silent=True, thread_count=1),
    func=tr.transform,
    inverse_func=tr.inverse_transform)

bc_lightgbm = TransformedTargetRegressor(
    regressor=LGBMRegressor(),
    func=tr.transform, inverse_func=tr.inverse_transform)

bc_xgboost = TransformedTargetRegressor(
    regressor=XGBRegressor(),
    func=tr.transform, inverse_func=tr.inverse_transform)

bc_lasso = TransformedTargetRegressor(
    regressor=Lasso(tol=1e-1, max_iter=1e4),
    func=tr.transform, inverse_func=tr.inverse_transform)

model = StackingRegressor(estimators=[
    ('bc_catboost', bc_catboost),
    ('bc_lightgbm', bc_lightgbm),
    ('bc_xgboost', bc_xgboost),
    ('bc_lasso', bc_lasso),
])

params = {
    'bc_catboost__regressor__n_estimators': IntUniformDistribution(100, 500),
    'bc_catboost__regressor__learning_rate': LogUniformDistribution(1e-3, 1e-1),
    'bc_catboost__regressor__l2_leaf_reg': LogUniformDistribution(1e-4, 1e-1),
    'bc_catboost__regressor__colsample_bylevel': UniformDistribution(0.1, 1.0),
    'bc_catboost__regressor__subsample': UniformDistribution(0.1, 1.0),
    'bc_catboost__regressor__max_depth': IntUniformDistribution(1, 8),
    'bc_catboost__regressor__min_data_in_leaf': IntUniformDistribution(1, 300),
    'bc_lightgbm__regressor__n_estimators': IntUniformDistribution(50, 500),
    'bc_lightgbm__regressor__max_depth': IntUniformDistribution(3, 1000),
    'bc_lightgbm__regressor__num_leaves': IntUniformDistribution(3, 1500),
    'bc_lightgbm__regressor__learning_rate': LogUniformDistribution(1e-4, 1e-1),
    'bc_lightgbm__regressor__subsample': UniformDistribution(0.1, 1.0),
    'bc_lightgbm__regressor__colsample_bytree': UniformDistribution(0.1, 1.0),
    'bc_lightgbm__regressor__colsample_bynode': UniformDistribution(0.1, 1.0),
    'bc_xgboost__regressor__n_estimators': IntUniformDistribution(50, 500),
    'bc_xgboost__regressor__max_depth': IntUniformDistribution(3, 1000),
    'bc_xgboost__regressor__max_leaves': IntUniformDistribution(3, 1500),
    'bc_xgboost__regressor__learning_rate': LogUniformDistribution(1e-4, 1e-1),
    'bc_xgboost__regressor__subsample': UniformDistribution(0.1, 1.0),
    'bc_xgboost__regressor__colsample_bytree': UniformDistribution(0.1, 1.0),
    'bc_xgboost__regressor__colsample_bylevel': UniformDistribution(0.1, 1.0),
    'bc_xgboost__regressor__colsample_bynode': UniformDistribution(0.1, 1.0),
    'bc_lasso__regressor__alpha': LogUniformDistribution(1e-6, 1e3),
}
