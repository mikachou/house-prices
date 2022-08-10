import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, ElasticNetCV
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from optuna.distributions import \
    CategoricalDistribution, IntUniformDistribution, \
    UniformDistribution, LogUniformDistribution

# bc_catboost
# log_xgboost
# bc_mlp
# log_svm
# log_lasso

train = pd.read_csv('./data/train.csv')

tr = PowerTransformer(method='box-cox')
tr.fit(train['SalePrice'].values.reshape(-1, 1))

bc_catboost = TransformedTargetRegressor(
    regressor=CatBoostRegressor(silent=True, thread_count=1, **{
        'n_estimators': 408,
        'learning_rate': 0.04617626777220529,
        'l2_leaf_reg': 0.000210494363264604,
        'colsample_bylevel': 0.7621275741308916,
        'subsample': 0.43994222377812675,
        'max_depth': 5,
        'min_data_in_leaf': 262
    }),
    func=tr.transform,
    inverse_func=tr.inverse_transform)

log_xgboost = TransformedTargetRegressor(
    regressor=XGBRegressor(**{
        'n_estimators': 481,
        'max_depth': 444,
        'max_leaves': 204,
        'learning_rate': 0.02783317711561734,
        'subsample': 0.22485696412550332,
        'colsample_bytree': 0.6747968891778797,
        'colsample_bylevel': 0.10128413882898236,
        'colsample_bynode': 0.737348115524531
    }),
    func=np.log1p, inverse_func=np.expm1)

bc_mlp = TransformedTargetRegressor(
    regressor=MLPRegressor(
        max_iter=1000, solver='sgd', learning_rate_init=1e-2, **{
            'hidden_layer_sizes': (4, 4),
            'alpha': 0.0007978091052275475
        }),
    func=tr.transform, inverse_func=tr.inverse_transform)

log_svm = TransformedTargetRegressor(
    regressor=SVR(**{
        'C': 1.9484177073961009,
        'gamma': 0.00013706545185322646
    }), func=np.log1p, inverse_func=np.expm1)

log_lasso = TransformedTargetRegressor(
    regressor=Lasso(tol=1e-1, max_iter=1e4, **{
        'alpha': 0.006567876651792052
    }), func=np.log1p, inverse_func=np.expm1)

model = StackingRegressor(estimators=[
    ('bc_catboost', bc_catboost),
    ('log_xgboost', log_xgboost),
    ('bc_mlp', bc_mlp),
    ('log_svm', log_svm),
    ('log_lasso', log_lasso),
], final_estimator=ElasticNetCV(cv=5))

params = {
    'final_estimator__l1_ratio': UniformDistribution(0.0, 1.0),
    'final_estimator__eps': LogUniformDistribution(1e-5, 1e0),
}
