import sys
import importlib
import numpy as np
import pandas as pd
from inc.search import search_cv

# import modules given CLI arguments
pmod = importlib.import_module('inc.preprocessor.' + sys.argv[1]) # e.g "preprocessor1"
mmod = importlib.import_module('inc.model.' + sys.argv[2]) # e.g "log_linear"

preprocessor = getattr(pmod, 'preprocessor')
model = getattr(mmod, 'model')
params = getattr(mmod, 'params')

# load datasets
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

pre = preprocessor(train, test)

X_train = pre.fit_transform(train.drop(['Id', 'SalePrice'], axis=1))
print(X_train.shape)
y_train = train['SalePrice']

n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 1

# search for optimal hypermarameters
scv = search_cv(X_train, y_train, model, params, n_trials=n_trials)
# reg.fit(X_train, y_train)

X_test = pre.transform(test.drop('Id', axis=1))
print(X_test.shape)
y_pred = scv.predict(X_test)

submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred})

# dirty trick : replace inf value with max value
m = submission.loc[submission['SalePrice'] != np.inf, 'SalePrice'].max()
submission['SalePrice'].replace(np.inf,m,inplace=True)

submission.to_csv('submissions/submission.csv', index=False)
