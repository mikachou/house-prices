"""Provide catboost regression
"""
from catboost import CatBoostRegressor
from optuna.distributions import \
    CategoricalDistribution, IntUniformDistribution, \
    UniformDistribution, LogUniformDistribution

params = {
    'iterations': IntUniformDistribution(100, 200),
    'learning_rate': LogUniformDistribution(0.03, 0.1),
    'depth': IntUniformDistribution(2, 8),
    'l2_leaf_reg': LogUniformDistribution(0.2, 3)
}

model = CatBoostRegressor()
