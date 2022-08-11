import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# bc_catboost
# log_xgboost
# bc_mlp
# log_svm
# log_lasso

train = pd.read_csv('./data/train.csv')

tr = PowerTransformer(method='box-cox')
tr.fit(train['SalePrice'].values.reshape(-1, 1))

def get_estimators():

    bc_catboost = TransformedTargetRegressor(
        regressor=CatBoostRegressor(
            silent=True, thread_count=1, n_estimators=1000, **{
                'learning_rate': 0.030449948522082177,
                'l2_leaf_reg': 0.035200852390827764,
                'colsample_bylevel': 0.43852407943493643,
                'subsample': 0.3933004151832541,
                'max_depth': 5,
                'min_data_in_leaf': 250,
            }),
        func=tr.transform,
        inverse_func=tr.inverse_transform)

    bc_xgboost = TransformedTargetRegressor(
        regressor=XGBRegressor(n_estimators=1000, **{
            'max_depth': 565,
            'max_leaves': 868,
            'learning_rate': 0.01069948249576208,
            'subsample': 0.16451988877705878,
            'colsample_bytree': 0.6306905781035185,
            'colsample_bylevel': 0.9136857442842662,
            'colsample_bynode': 0.9249030949751555,
        }),
        func=np.log1p, inverse_func=np.expm1)

    bc_mlp = TransformedTargetRegressor(
        regressor=MLPRegressor(
            max_iter=1000, solver='sgd', learning_rate_init=1e-2, **{
                'hidden_layer_sizes': (4, 4),
                'alpha': 0.0007978091052275475,
            }),
        func=tr.transform, inverse_func=tr.inverse_transform)

    log_svm = TransformedTargetRegressor(
        regressor=SVR(**{
            'C': 1.9484177073961009,
            'gamma': 0.00013706545185322646,
        }), func=np.log1p, inverse_func=np.expm1)

    log_lasso = TransformedTargetRegressor(
        regressor=Lasso(tol=1e-1, max_iter=1e4, **{
            'alpha': 0.006567876651792052,
        }), func=np.log1p, inverse_func=np.expm1)

    return [
        ('bc_catboost', bc_catboost),
        ('bc_xgboost', bc_xgboost),
        ('bc_mlp', bc_mlp),
        ('log_svm', log_svm),
        ('log_lasso', log_lasso),
    ]