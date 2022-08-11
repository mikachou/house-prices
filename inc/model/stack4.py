"""Try blending all models including stacking ones
Does not work :( needs improvements
"""
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from xgboost import XGBRegressor
from inc.model.base import stack3

base_estimators = stack3.get_estimators()

stack_3_1 = StackingRegressor(
    estimators=base_estimators, final_estimator=ElasticNetCV(cv=5, **{
        'l1_ratio': 0.8712664510237272,
        'eps': 1.6136134823835553e-05
    }))

stack_3_3 = StackingRegressor(
    estimators=base_estimators, final_estimator=XGBRegressor(
        n_estimators=1000,
        **{
            'max_depth': 752,
            'max_leaves': 564,
            'learning_rate': 0.007056121928922043,
            'subsample': 0.8448512330295146,
            'colsample_bytree': 0.9910007579688851,
            'colsample_bylevel': 0.6843120886179166,
            'colsample_bynode': 0.2544875308156066
        }))

estimators = base_estimators \
    + [('stack_3_1', stack_3_1), ('stack_3_3', stack_3_3)]

model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(fit_intercept=False))

params = {}