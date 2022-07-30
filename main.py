import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures, OneHotEncoder, FunctionTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.base import TransformerMixin, BaseEstimator

# load datasets
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

num_cols = [col for col in train.columns if train.dtypes[col] != 'object']
num_cols.remove('SalePrice')
num_cols.remove('Id')
num_cols.remove('MSSubClass') # MSSubClass is actually categorical
cat_cols = [col for col in train.columns if train.dtypes[col] == 'object']
cat_cols.append('MSSubClass')

print(num_cols)
print(cat_cols)

missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
#missing.plot.bar()
#plt.show(block=True)
print(missing)

class DataframeFunctionTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self

    def predict(self, X, y=None, **fit_params):
        return self.fit_transform(X, y=None, **fit_params)

class DenseTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray()

    def predict(self, X, y=None, **fit_params):
        return self.fit_transform(X, y=None, **fit_params)

class EmbedTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None, **fit_params):
        self.transformer.fit(X)

        return self
    
    def transform(self, X, y=None, **fit_params):
        return self.transformer.transform(X)

    def predict(self, X, y=None, **fit_params):
        return self.fit_transform(X, y=None, **fit_params)

def fillna(df):
    copy = df.copy()
    values = {
        'Electrical': 'SBrkr',
        'MasVnrType': 'None',
        'MasVnrArea': 0,
    }

    for k, v in values.items():
        copy[k].fillna(v, inplace=True)

    copy['GarageYrBlt'].fillna(copy['YearBuilt'], inplace=True)

    for k in ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtExposure',
        'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual',
        'GarageCond', 'FireplaceQu', 'Fence', 'Alley', 'MiscFeature',
        'PoolQC']:
        copy[k].fillna('NA', inplace=True)

    return copy

def bool_features(df):
    copy = df.copy()

    copy['HasBasement'] = copy['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    copy['HasGarage'] = copy['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    copy['Has2ndFloor'] = copy['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    copy['HasMasVnr'] = copy['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
    copy['HasWoodDeck'] = copy['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
    copy['HasPorch'] = copy['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
    copy['HasPool'] = copy['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    copy['IsNew'] = copy['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

    return copy

print()

missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
#missing.plot.bar()
#plt.show(block=True)
print(missing)

ord_scores = ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
scores_cat_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
    'GarageCond', 'PoolQC']

other_cat_cols = list(
    set(cat_cols)
    - set(scores_cat_cols) 
    - set(['Electrical', 'CentralAir', 'GarageFinish', 'PavedDrive']))

log_num_cols = [
    'GrLivArea', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF',
    'LotArea', 'LotFrontage', 'KitchenAbvGr', 'GarageArea'
]

quad_num_cols = [
    'OverallQual', 'YearBuilt', 'YearRemodAdd',
    '2ndFlrSF', 'GrLivArea',
]

columns = ColumnTransformer(transformers = [
    ('scores_transform', EmbedTransformer(OrdinalEncoder(categories=len(scores_cat_cols) * [ord_scores], handle_unknown='use_encoded_value', unknown_value=-1)), scores_cat_cols),
    ('log_transform', FunctionTransformer(np.log1p), log_num_cols),
    ('quad_transform', EmbedTransformer(PolynomialFeatures(degree=2)), quad_num_cols),
    ('electrical', EmbedTransformer(OrdinalEncoder(categories=[['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr']])), ['Electrical']),
    ('central_air', EmbedTransformer(OrdinalEncoder(categories=[['N', 'Y']])), ['CentralAir']),
    ('garage_finish', EmbedTransformer(OrdinalEncoder(categories=[['NA', 'Unf', 'RFn', 'Fin']])), ['GarageFinish']),
    ('paved_drive', EmbedTransformer(OrdinalEncoder(categories=[['N', 'P', 'Y']], handle_unknown='use_encoded_value', unknown_value=-1)), ['PavedDrive']),
    ('one_hot', OneHotEncoder(handle_unknown='ignore'), other_cat_cols),
])

pipe = Pipeline([
    ('fillna', DataframeFunctionTransformer(fillna)),
    ('bool_features', DataframeFunctionTransformer(bool_features)),
    ('columns', columns),
    ('to_dense', DenseTransformer()),
    ('impute', EmbedTransformer(KNNImputer())), # actually only LotFrontage remaining
    ('reg', TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1)),
])

X = train.drop(['Id', 'SalePrice'], axis=1)
y = train['SalePrice']

pipe.fit(X, y)
y_pred = pipe.predict(test.drop('Id', axis=1))

submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred})

# dirty trick : replace inf value with max value
m = submission.loc[submission['SalePrice'] != np.inf, 'SalePrice'].max()
submission['SalePrice'].replace(np.inf,m,inplace=True)

submission.to_csv('submissions/submission.csv', index=False)
