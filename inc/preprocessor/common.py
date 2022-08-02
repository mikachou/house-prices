"""
Provides some utilities to preprocessors
"""

from sklearn.base import TransformerMixin, BaseEstimator


class DataframeFunctionTransformer(TransformerMixin, BaseEstimator):
    """
    Do transformations regarding a specific function
    """
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        """
        performs the function transformation
        """
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        """
        fit method
        """
        return self

class DenseTransformer(TransformerMixin, BaseEstimator):
    """
    Transforms parse matrix into numpy array
    """
    def fit(self, X, y=None, **fit_params):
        """
        fit method
        """
        return self

    def transform(self, X, y=None, **fit_params):
        """
        transform method
        """
        return X.toarray()

def fillna(df):
    """fill na values on dataframe

    Args:
        df (pandas.DataFrame): original dataframe
    Returns:
        pandas.DataFrame: copy of original dataframe with filled values
    """
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
    """Create bool features for some variables

    Args:
        df (pandas.DataFrame): dataframe

    Returns:
        pandas.DataFrame: copy of original dataframe with added bool features
    """
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
