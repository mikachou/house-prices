"""Outliers processing
"""
def remove_outliers(train):
    """Remove outliers from trainset

    Args:
        train (pandas.Dataframe): train set
    """

    # get Ids after EDA
    # execute eda_outliers.py for outliers analysis

    ids = [
        692, 1183,  # SalePrice > 700000
        935,        # LotFrontage > 300
        1299,       # BsmtFinSF1 > 4000, TotalBsmtSF > 6000, 
                    # 1stFlrSF > 4000, GrLivArea > 5000
        323,        # BsmtFinSF2 > 1400
        198,        # EnclosedPorch > 400
        347, 1231   # MiscVal > 5000
    ]

    return train[~train['Id'].isin(ids)]
