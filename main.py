import sys
import importlib
import numpy as np
import pandas as pd

# import modules given CLI arguments
pm = importlib.import_module('inc.preprocessor.' + sys.argv[1]) # e.g "preprocessor1"
mm = importlib.import_module('inc.model.' + sys.argv[2]) # e.g "log_linear"

preprocessor = getattr(pm, 'preprocessor')
model = getattr(mm, 'model')

# load datasets
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

pre = preprocessor(train, test)
reg = model()

X_train = pre.fit_transform(train.drop(['Id', 'SalePrice'], axis=1))
print(X_train.shape)
y_train = train['SalePrice']

reg.fit(X_train, y_train)

X_test = pre.transform(test.drop('Id', axis=1))
print(X_test.shape)
y_pred = reg.predict(X_test)

submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred})

# dirty trick : replace inf value with max value
m = submission.loc[submission['SalePrice'] != np.inf, 'SalePrice'].max()
submission['SalePrice'].replace(np.inf,m,inplace=True)

submission.to_csv('submissions/submission.csv', index=False)
