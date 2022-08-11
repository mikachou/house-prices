"""Dummy blend without cross-validation
"""
import sys
from datetime import datetime
import importlib
import numpy as np
import pandas as pd
from inc.search import search_cv
from inc.outliers.outliers1 import remove_outliers
from inc.preprocessor.preprocessor1 import preprocessor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from inc.model.base import stack3

# initialise numpy random seed
np.random.seed(314)

# load datasets
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# remove outliers
print(train.shape)
train = remove_outliers(train)
print(train.shape)

# preprocessing

pre = preprocessor(train, test)

X_train = pre.fit_transform(train.drop(['Id', 'SalePrice'], axis=1))
print(X_train.shape)
y_train = train['SalePrice']

# estimators

base_estimators = stack3.get_estimators()

# queue stacking model in the end of estimators

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

# create linear model with pre-fit estimators

# starting scikit-learn 1.1 it seems possible to create blending
# models thanks to prefit estimators and cv argument set as 'prefit'
# however, it doesn't work with the version of scikit-learn used
# for this project (1.0.2)

# # fit estimators with preprocessed data
# for estimator in estimators:
#     print(f'fit {estimator[0]}')
#     estimator[1].fit(X_train, y_train)

# blend = StackingRegressor(
#     estimators=estimators,
#     final_estimator=LinearRegression(),
#     cv='prefit')

# blend.fit(X_train, y_train)
#
# X_test = pre.transform(test.drop('Id', axis=1))
# print(X_test.shape)
# y_pred = blend.predict(X_test)

# because of scikit-learn < 1.1 :@, we build a trainer for linear model

# actually must implement a cross-validation...
for estimator in estimators:
    print(f'fit {estimator[0]}')
    estimator[1].fit(X_train, y_train)

y_pred_on_train = pd.DataFrame()
for estimator in estimators:
    print(f'predict {estimator[0]}')
    y_pred_on_train[f'y_{estimator[0]}'] = estimator[1].predict(X_train)

blend = LinearRegression(fit_intercept=False)
blend.fit(y_pred_on_train, y_train)

print(blend.coef_, blend.intercept_)

y_pred_on_train = pd.DataFrame()
for estimator in estimators:
    print(f'predict {estimator[0]}')
    y_pred_on_train[f'y_{estimator[0]}'] = estimator[1].predict(X_train)

X_test = pre.transform(test.drop('Id', axis=1))
print(X_test.shape)

y_pred_on_test = pd.DataFrame()
for estimator in estimators:
    print(f'predict {estimator[0]}')
    y_pred_on_test[f'y_{estimator[0]}'] = estimator[1].predict(X_test)

y_pred = blend.predict(y_pred_on_test)
print(test.shape, y_pred.shape)

submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred})

# dirty trick : replace inf value with max value
m = submission.loc[submission['SalePrice'] != np.inf, 'SalePrice'].max()
submission['SalePrice'].replace(np.inf,m,inplace=True)

base_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

submission.to_csv(f'submissions/{base_name}.csv', index=False)
