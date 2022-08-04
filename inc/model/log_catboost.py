"""log catboost regression
"""
import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor
from optuna.distributions import \
    CategoricalDistribution, IntUniformDistribution, \
    UniformDistribution, LogUniformDistribution

params = {
    'regressor__iterations': IntUniformDistribution(100, 200),
    'regressor__learning_rate': LogUniformDistribution(0.03, 0.1),
    'regressor__depth': IntUniformDistribution(2, 8),
    'regressor__l2_leaf_reg': LogUniformDistribution(0.2, 3)
}

model = TransformedTargetRegressor(
        regressor=CatBoostRegressor(), func=np.log1p, inverse_func=np.expm1)