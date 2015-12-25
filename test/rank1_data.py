
import numpy as np


def create_rank1_dataset(n=5, fraction_missing=0.1, symmetric=False):
    np.random.seed(0)
    x = np.random.randn(n)
    y = np.random.randn(n)

    XY = np.outer(x, y)

    if symmetric:
        XY = 0.5 * XY + 0.5 * XY.T

    missing_raw_values = np.random.uniform(0, 1, (n, n))
    missing_mask = missing_raw_values < fraction_missing

    XY_incomplete = XY.copy()
    # fill missing entries with NaN
    XY_incomplete[missing_mask] = np.nan

    return XY, XY_incomplete, missing_mask
