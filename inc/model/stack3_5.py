"""Provides stacking regressor
"""
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from inc.model.base import stack3

estimators = stack3.get_estimators()

model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(fit_intercept=False))

params = {}
