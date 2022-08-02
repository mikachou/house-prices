"""
Provide feature preprocessor pipeline
"""

import numpy as np
from sklearn.preprocessing import \
    OrdinalEncoder, PolynomialFeatures, OneHotEncoder, FunctionTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from inc.preprocessor.common import * # pylint: disable=wildcard-import, unused-wildcard-import

def preprocessor(train, test):
    """Provide features preprocessor pipeline

    Args:
        train (pandas.DataFrame): train set
        test (pandas.DataFrame): test set

    Returns:
        sklearn.pipeline.Pipeline: pipeline
    """
    num_cols = [col for col in train.columns if train.dtypes[col] != 'object']
    num_cols.remove('SalePrice')
    num_cols.remove('Id')
    num_cols.remove('MSSubClass') # MSSubClass is actually categorical
    cat_cols = [col for col in train.columns if train.dtypes[col] == 'object']
    cat_cols.append('MSSubClass')

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
        ('scores_transform', OrdinalEncoder(
            categories=len(scores_cat_cols) * [ord_scores],
            handle_unknown='use_encoded_value', unknown_value=-1), scores_cat_cols),
        ('log_transform', FunctionTransformer(np.log1p), log_num_cols),
        ('quad_transform', PolynomialFeatures(degree=2), quad_num_cols),
        ('electrical', OrdinalEncoder(
            categories=[['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr']]), ['Electrical']),
        ('central_air', OrdinalEncoder(
            categories=[['N', 'Y']]), ['CentralAir']),
        ('garage_finish', OrdinalEncoder(
            categories=[['NA', 'Unf', 'RFn', 'Fin']]), ['GarageFinish']),
        ('paved_drive', OrdinalEncoder(
            categories=[['N', 'P', 'Y']],
            handle_unknown='use_encoded_value', unknown_value=-1), ['PavedDrive']),
        ('one_hot', OneHotEncoder(handle_unknown='ignore'), other_cat_cols),
    ])

    pre = Pipeline([
        ('fillna', DataframeFunctionTransformer(fillna)),
        ('bool_features', DataframeFunctionTransformer(bool_features)),
        ('columns', columns),
        ('to_dense', DenseTransformer()),
        ('impute', KNNImputer()), # actually only LotFrontage remaining
    ])

    return pre
