from inc.model.base import stack3
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNetCV
from optuna.distributions import \
    CategoricalDistribution, IntUniformDistribution, \
    UniformDistribution, LogUniformDistribution

# bc_catboost
# log_xgboost
# bc_mlp
# log_svm
# log_lasso

estimators = stack3.get_estimators()

model = StackingRegressor(
    estimators=estimators, final_estimator=ElasticNetCV(cv=5))

params = {
    'final_estimator__l1_ratio': UniformDistribution(0.0, 1.0),
    'final_estimator__eps': LogUniformDistribution(1e-5, 1e0),
}
