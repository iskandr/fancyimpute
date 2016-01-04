import numpy as np


def reconstruction_error(XY, XY_completed, missing_mask, name=None):
    """
    Returns mean squared error and mean absolute error for
    completed matrices.
    """
    value_pairs = [
        (i, j, XY[i, j], XY_completed[i, j])
        for i in range(XY.shape[0])
        for j in range(XY.shape[1])
        if missing_mask[i, j]
    ]
    print("First 10 reconstructed values:")
    for (i, j, x, xr) in value_pairs[:10]:
        print("  (%d,%d)  %0.4f ~= %0.4f" % (i, j, x, xr))
    diffs = [actual - predicted for (_, _, actual, predicted) in value_pairs]
    missing_mse = np.mean([diff ** 2 for diff in diffs])
    missing_mae = np.mean([np.abs(diff) for diff in diffs])
    print("%sMSE: %0.4f, MAE: %0.4f" % (
        "" if not name else name + " ",
        missing_mse,
        missing_mae))
    return missing_mse, missing_mae
