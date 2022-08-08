from inc.preprocessor.common.preprocessor import common_preprocessor
from sklearn.impute import KNNImputer

def preprocessor(train, test):
    pre = common_preprocessor(train, test)

    pre.steps.append(('impute', KNNImputer()))

    return pre
