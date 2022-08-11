"""hyperparameters optimization with OptunaSearchCV

Returns:
    optuna.integration.OptunaSearchCV: prefit model after hyperparameters tuning
    float: time to execute the optimization in seconds
"""
import time
from optuna.integration import OptunaSearchCV

def search_cv(X, y, model, param_distributions={}, random_state=None, # pylint: disable=dangerous-default-value too-many-arguments
    n_splits=5, n_jobs=-1, n_trials=10, scoring='neg_mean_squared_log_error'):
    """fit search_cv after hyperparameters tuning

    Args:
        X (numpy.array): data
        y (numpy.array): target
        model (object): sklearn model
        param_distributions (dict, optional): hyperparameters to optimize. Defaults to {}.
        random_state (int, optional): random state. Defaults to None.
        n_splits (int, optional): number of CV folds. Defaults to 5.
        n_jobs (int, optional): number of jobs (-1 = all cpu cores). Defaults to -1.
        n_trials (int, optional): number of trials for hyperparameters tuning. Defaults to 10.
        scoring (str, optional): score to maximize. Defaults to 'neg_mean_squared_log_error'.

    Returns:
        _type_: _description_
    """
    search_args = {
        'estimator': model,
        'param_distributions': param_distributions,
        'cv': n_splits,
        'scoring': scoring,
        'n_jobs': n_jobs,
        'n_trials': n_trials,
        'random_state': random_state,
    }

    opt_search_cv = OptunaSearchCV(**search_args)

    start = time.time()
    opt_search_cv.fit(X, y.values.ravel())
    end = time.time()

    print('time:', end - start, 'seconds')

    return opt_search_cv, end - start
