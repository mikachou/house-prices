"""log catboost regression
"""
import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor
from optuna.distributions import \
    IntUniformDistribution, UniformDistribution, LogUniformDistribution

params = {
    # 'regressor__n_estimators': IntUniformDistribution(100, 500),
    'regressor__learning_rate': LogUniformDistribution(1e-3, 1e-1),
    'regressor__l2_leaf_reg': LogUniformDistribution(1e-4, 1e-1),
    'regressor__colsample_bylevel': UniformDistribution(0.1, 1.0),
    'regressor__subsample': UniformDistribution(0.1, 1.0),
    'regressor__max_depth': IntUniformDistribution(1, 8),
    'regressor__min_data_in_leaf': IntUniformDistribution(1, 300),
}

model = TransformedTargetRegressor(
    regressor=CatBoostRegressor(
        silent=True, thread_count=1, n_estimators=1000),
    func=np.log1p,
    inverse_func=np.expm1)
