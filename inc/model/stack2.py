"""Stacking regressor model
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingRegressor

train = pd.read_csv('./data/train.csv')

tr = PowerTransformer(method='box-cox')
tr.fit(train['SalePrice'].values.reshape(-1, 1))

bc_catboost = TransformedTargetRegressor(
    regressor=CatBoostRegressor(
        silent=True, thread_count=1, n_estimators=379,
        learning_rate=0.04238134383495472,
        l2_leaf_reg=0.01885700551331138,
        colsample_bylevel=0.11010653142053686,
        subsample=0.7165173672039271, max_depth=5,
        min_data_in_leaf=229
    ),
    func=tr.transform,
    inverse_func=tr.inverse_transform)

log_catboost = TransformedTargetRegressor(
    regressor=CatBoostRegressor(
        silent=True, thread_count=1, n_estimators=425,
        learning_rate=0.05701026089135576,
        l2_leaf_reg=0.0036589644719517843,
        colsample_bylevel=0.13306451666681168,
        subsample=0.5147762360352649, max_depth=4,
        min_data_in_leaf=61
    ),
    func=np.log1p, inverse_func=np.expm1)

bc_lightgbm = TransformedTargetRegressor(
    regressor=LGBMRegressor(**{
        'n_estimators': 438, 'max_depth': 479, 'num_leaves': 115,
        'learning_rate': 0.02144266891698913,
        'subsample': 0.8972241651069026,
        'colsample_bytree': 0.23781296789020387,
        'colsample_bynode': 0.1721610185300597}),
    func=tr.transform, inverse_func=tr.inverse_transform)

log_lightgbm = TransformedTargetRegressor(
    regressor=LGBMRegressor(**{
        'n_estimators': 292, 'max_depth': 426, 'num_leaves': 1076,
        'learning_rate': 0.022963548887449958,
        'subsample': 0.8741055289371807,
        'colsample_bytree': 0.5263817655194819,
        'colsample_bynode': 0.14940137632697237}),
    func=np.log1p, inverse_func=np.expm1)

bc_xgboost = TransformedTargetRegressor(
    regressor=XGBRegressor(**{
        'n_estimators': 363, 'max_depth': 806, 'max_leaves': 580,
        'learning_rate': 0.024103398506205836,
        'subsample': 0.171547233795194,
        'colsample_bytree': 0.8062248075428511,
        'colsample_bylevel': 0.5746911154580273,
        'colsample_bynode': 0.5883060694538585
    }),
    func=tr.transform, inverse_func=tr.inverse_transform)

log_xgboost = TransformedTargetRegressor(
    regressor=XGBRegressor(**{
        'n_estimators': 423, 'max_depth': 538, 'max_leaves': 406,
        'learning_rate': 0.02697471723450166,
        'subsample': 0.5631332943184617,
        'colsample_bytree': 0.5230934838948597,
        'colsample_bylevel': 0.3567056193854949,
        'colsample_bynode': 0.39002260127266114
    }),
    func=np.log1p, inverse_func=np.expm1)

bc_lasso = TransformedTargetRegressor(
    regressor=Lasso(tol=1e-1, max_iter=1e4, alpha=0.01635763009773209),
    func=tr.transform, inverse_func=tr.inverse_transform)

log_lasso = TransformedTargetRegressor(
    regressor=Lasso(tol=1e-1, max_iter=1e4, alpha=0.006877314823517839),
    func=np.log1p, inverse_func=np.expm1)

model = StackingRegressor(estimators=[
    ('bc_catboost', bc_catboost),
    ('log_catboost', log_catboost),
    ('bc_lightgbm', bc_lightgbm),
    ('log_lightgbm', log_lightgbm),
    ('bc_xgboost', bc_xgboost),
    ('log_xgboost', log_xgboost),
    ('bc_lasso', bc_lasso),
    ('log_lasso', log_lasso),
])

params = {}
