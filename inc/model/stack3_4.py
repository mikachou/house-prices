import numpy as np
from inc.model.base import stack3
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from optuna.distributions import \
    CategoricalDistribution, IntUniformDistribution, \
    UniformDistribution, LogUniformDistribution

# bc_catboost
# bc_xgboost
# bc_mlp
# log_svm
# log_lasso

estimators = stack3.get_estimators()

model = StackingRegressor(
    estimators=estimators,
        final_estimator=TransformedTargetRegressor(
        regressor=XGBRegressor(n_estimators=1000),
        func=np.log1p, inverse_func=np.expm1))

params = {
    # 'final_estimator__n_estimators': IntUniformDistribution(50, 1500),
    'final_estimator__regressor__max_depth': IntUniformDistribution(3, 1000),
    'final_estimator__regressor__max_leaves': IntUniformDistribution(3, 1500),
    'final_estimator__regressor__learning_rate': LogUniformDistribution(1e-4, 1e-1),
    'final_estimator__regressor__subsample': UniformDistribution(0.1, 1.0),
    'final_estimator__regressor__colsample_bytree': UniformDistribution(0.1, 1.0),
    'final_estimator__regressor__colsample_bylevel': UniformDistribution(0.1, 1.0),
    'final_estimator__regressor__colsample_bynode': UniformDistribution(0.1, 1.0),
}
