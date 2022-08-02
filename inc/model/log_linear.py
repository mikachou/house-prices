"""Provide log transformed linear regression
"""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression

def model():
    """Define the model

    Returns:
        sklearn.compose.TransformedTargetRegressor: the model
    """
    return TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1)
