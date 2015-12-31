
import numpy as np


def create_rank_k_dataset(
        n_rows=5,
        n_cols=5,
        k=3,
        fraction_missing=0.1,
        symmetric=False,
        random_seed=0):
    np.random.seed(random_seed)
    x = np.random.randn(n_rows, k)
    y = np.random.randn(k, n_cols)

    XY = np.dot(x, y)

    if symmetric:
        assert n_rows == n_cols
        XY = 0.5 * XY + 0.5 * XY.T

    missing_raw_values = np.random.uniform(0, 1, (n_rows, n_cols))
    missing_mask = missing_raw_values < fraction_missing

    XY_incomplete = XY.copy()
    # fill missing entries with NaN
    XY_incomplete[missing_mask] = np.nan

    return XY, XY_incomplete, missing_mask


# create some default data to be shared across tests
XY, XY_incomplete, missing_mask = create_rank_k_dataset(
    n_rows=500,
    n_cols=10,
    k=3,
    fraction_missing=0.25)
