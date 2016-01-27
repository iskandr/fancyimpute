import numpy as np
from nose.tools import eq_

from fancyimpute import SimilarityWeightedAveraging


def test_similarity_weighted_column_averaging():
    X = np.array([
        [0.1, 0.9, 0.2],
        [0.8, 0.1, 0.01],
        [0.95, 0.2, 0.3],
        [0.14, 0.85, 0.3],
    ])
    X_incomplete = X.copy()
    X_incomplete[1, 1] = np.nan
    X_incomplete[3, 0] = np.nan
    missing_mask = np.isnan(X_incomplete)

    solver = SimilarityWeightedAveraging()
    X_filled = solver.complete(X_incomplete)
    eq_(X_incomplete.shape, X_filled.shape)
    diff = (X - X_filled)[missing_mask]
    abs_diff = np.abs(diff)
    mae = np.mean(abs_diff)
    print("MAE", mae)
    assert mae < 0.1, "Difference between imputed values! MAE=%0.4f" % mae

if __name__ == "__main__":
    test_similarity_weighted_column_averaging()
