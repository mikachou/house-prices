"""data preprocessor

Returns:
    scikit-learn.pipeline.Pipeline: pipeline
"""
from sklearn.impute import SimpleImputer
from inc.preprocessor.common.preprocessor import common_preprocessor

def preprocessor(train, test):
    """data preprocessor

    Returns:
        scikit-learn.pipeline.Pipeline: pipeline
    """
    pre = common_preprocessor(train, test)

    pre.steps.append(('impute', SimpleImputer(strategy='mean')))

    return pre
