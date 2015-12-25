import numpy as np
from fancyimpute import BayesianRegression
from sklearn.linear_model import Ridge


def test_brr_like_sklearn():
    n = 10000
    d = 10
    sigma_sqr = 5
    X = np.random.randn(n, d)
    beta_true = np.random.random(d)
    y = np.dot(X, beta_true) + np.sqrt(sigma_sqr) * np.random.randn(n)
    X_tr = X[:n / 2, :]
    y_tr = y[:n / 2]
    X_ts = X[n / 2:, :]
    y_ts = y[n / 2:]

    # prediction with my own bayesian ridge
    br = BayesianRegression()
    br.fit(X_tr, y_tr)
    y_ts_br = br.predict(X_ts)
    br_error = np.mean(np.abs(y_ts_br - y_ts))

    # let's compare to scikit-learn's ridge regression
    rr = Ridge(1e-5)
    rr.fit(X_tr, y_tr)
    y_ts_rr = rr.predict(X_ts)
    rr_error = np.mean(np.abs(y_ts_rr - y_ts))

    assert br_error - rr_error < 0.1, \
        "Error is significantly worse than sklearn's ridge regression."
