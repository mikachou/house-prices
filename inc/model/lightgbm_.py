"""Provide lightgbm regression
"""
from lightgbm import LGBMRegressor
from optuna.distributions import \
    IntUniformDistribution, UniformDistribution, LogUniformDistribution

params = {
    'n_estimators': IntUniformDistribution(50, 500),
    'max_depth': IntUniformDistribution(3, 1000),
    'num_leaves': IntUniformDistribution(3, 1500),
    'learning_rate': LogUniformDistribution(1e-4, 1e-1),
    'subsample': UniformDistribution(0.1, 1.0),
    'colsample_bytree': UniformDistribution(0.1, 1.0),
    'colsample_bynode': UniformDistribution(0.1, 1.0),
}

model = LGBMRegressor()
