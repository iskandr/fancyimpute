import numpy as np
from nose.tools import eq_

from fancyimpute.dense_knn import (knn_impute, knn_impute_experimental)
from low_rank_data import XY_incomplete, missing_mask


def test_knn_fast_gives_same_values_as_reference():
    X_filled_reference = knn_impute(XY_incomplete, missing_mask, k=3)
    X_filled2 = knn_impute_experimental(XY_incomplete, missing_mask, k=3)
    eq_(X_filled_reference.shape, X_filled2.shape)
    diff = X_filled_reference - X_filled2
    abs_diff = np.abs(diff)
    mae = np.mean(abs_diff)
    assert mae < 0.1, "Difference between imputed values! MAE=%0.4f" % mae
