"""Provide catboost regression
"""
from catboost import CatBoostRegressor
from optuna.distributions import \
    CategoricalDistribution, IntUniformDistribution, \
    UniformDistribution, LogUniformDistribution

params = {
    'n_estimators': IntUniformDistribution(100, 1000),
    'learning_rate': LogUniformDistribution(1e-3, 1e-1),
    'l2_leaf_reg': LogUniformDistribution(1e-4, 1e-1),
    'colsample_bylevel': UniformDistribution(0.1, 1.0),
    'subsample': UniformDistribution(0.1, 1.0),
    'max_depth': IntUniformDistribution(1, 8),
    'min_data_in_leaf': IntUniformDistribution(1, 300),
}

model = CatBoostRegressor(silent=True, thread_count=1)
