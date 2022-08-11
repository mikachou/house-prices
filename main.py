"""main scripts
"""
import sys
from datetime import datetime
import importlib
import numpy as np
import pandas as pd
from inc.search import search_cv

# initialise numpy random seed
np.random.seed(314)

# import modules given CLI arguments
omod = importlib.import_module('inc.outliers.' + sys.argv[1]) # e.g "outliers0"
pmod = importlib.import_module('inc.preprocessor.' + sys.argv[2]) # e.g "preprocessor1"
mmod = importlib.import_module('inc.model.' + sys.argv[3]) # e.g "log_linear"

remove_outliers = getattr(omod, 'remove_outliers')
preprocessor = getattr(pmod, 'preprocessor')
model = getattr(mmod, 'model')
params = getattr(mmod, 'params')

# load datasets
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# remove outliers
print(train.shape)
train = remove_outliers(train)
print(train.shape)

pre = preprocessor(train, test)

X_train = pre.fit_transform(train.drop(['Id', 'SalePrice'], axis=1))
print(X_train.shape)
y_train = train['SalePrice']

n_trials = int(sys.argv[4]) if len(sys.argv) > 4 else 1 # pylint: disable=invalid-name

# search for optimal hypermarameters
scv, time = search_cv(X_train, y_train, model, params, n_trials=n_trials)
# reg.fit(X_train, y_train)

print(scv.best_params_)

X_test = pre.transform(test.drop('Id', axis=1))
print(X_test.shape)
y_pred = scv.predict(X_test)

submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred})

# dirty trick : replace inf value with max value
m = submission.loc[submission['SalePrice'] != np.inf, 'SalePrice'].max()
submission['SalePrice'].replace(np.inf,m,inplace=True)

base_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

submission.to_csv(f'submissions/{base_name}.csv', index=False)

# log parameters
log_name = 'logs/' + base_name + '.log'
with open(log_name, 'w', encoding='utf8') as log:
    log.writelines([
        f'outliers: {sys.argv[1]}\n',
        f'preprocessor: {sys.argv[2]}\n',
        f'model: {sys.argv[3]}\n',
        f'trials: {sys.argv[4]}\n',
        f'best params: {scv.best_params_}\n',
        f'score: {scv.best_score_}\n',
        f'time: {time}\n',
    ])
