import numpy as np
from fancyimpute import BayesianRidgeRegression
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
    #  y_ts = y[n / 2:]

    # prediction with my own bayesian ridge
    lambda_reg = 1
    brr = BayesianRidgeRegression(lambda_reg,
                                  add_ones=True,
                                  normalize_lambda=False)
    brr.fit(X_tr, y_tr)
    y_ts_brr = brr.predict(X_ts)

    # let's compare to scikit-learn's ridge regression
    rr = Ridge(lambda_reg)
    rr.fit(X_tr, y_tr)
    y_ts_rr = rr.predict(X_ts)

    assert np.mean(np.abs(y_ts_brr - y_ts_rr)) < 0.001, \
        "Predictions are different from sklearn's ridge regression."

if __name__ == "__main__":
    test_brr_like_sklearn()
