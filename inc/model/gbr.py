"""Provide Gradient Boosting Regressor
"""
from sklearn.ensemble import GradientBoostingRegressor
from optuna.distributions import IntUniformDistribution, UniformDistribution

params = {
    'learning_rate': UniformDistribution(1e-3, 1.0),
    'n_estimators': IntUniformDistribution(50, 500),
    'subsample': UniformDistribution(0.1, 1.0),
    'max_depth': IntUniformDistribution(3, 300),
    'min_samples_split': IntUniformDistribution(3, 300),
    'min_samples_leaf': IntUniformDistribution(3, 300),
    'max_features': UniformDistribution(0.1, 1.0),
    'max_leaf_nodes': IntUniformDistribution(3, 300),
}

model = GradientBoostingRegressor()
