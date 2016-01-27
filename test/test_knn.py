import numpy as np
from nose.tools import eq_

from fancyimpute.knn_helpers import (
    knn_impute_few_observed,
    knn_impute_with_argpartition,
    knn_impute_optimistic,
    knn_impute_reference,
)
from low_rank_data import XY_incomplete, missing_mask


def _knn_implementation(impute_fn):
    X_filled_reference = knn_impute_reference(XY_incomplete.copy(), missing_mask, k=3)
    X_filled_other = impute_fn(XY_incomplete.copy(), missing_mask, k=3)
    eq_(X_filled_reference.shape, X_filled_other.shape)
    diff = X_filled_reference - X_filled_other
    abs_diff = np.abs(diff)
    mae = np.mean(abs_diff)
    assert mae < 0.1, \
        "Difference between imputed values! MAE=%0.4f, 1st rows: %s vs. %s" % (
            mae,
            X_filled_reference[0],
            X_filled_other[0]
        )


def test_knn_argpartition_same_as_reference():
    _knn_implementation(knn_impute_with_argpartition)


def test_knn_optimistic_same_as_reference():
    _knn_implementation(knn_impute_optimistic)


def test_knn_optimistic_few_observed():
    _knn_implementation(knn_impute_few_observed)
