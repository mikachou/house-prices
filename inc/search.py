import time
from optuna.integration import OptunaSearchCV

def search_cv(X, y, model, param_distributions={}, random_state=314,
    n_splits=5, n_jobs=-1, n_trials=10, scoring='neg_mean_squared_log_error'):
    search_args = {
        'estimator': model,
        'param_distributions': param_distributions,
        'cv': n_splits,
        'scoring': scoring,
        'n_jobs': n_jobs,
        'n_trials': n_trials,
        'random_state': random_state,
    }

    search_cv = OptunaSearchCV(**search_args)

    start = time.time()
    search_cv.fit(X, y.values.ravel())
    end = time.time()

    print('time:', end - start, 'seconds')

    return search_cv
