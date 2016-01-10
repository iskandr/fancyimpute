from time import time
import numpy as np
from nose.tools import eq_

from fancyimpute.normalized_distance import (
    all_pairs_normalized_distances,
    all_pairs_normalized_distances_reference
)
from low_rank_data import XY_incomplete


def test_normalized_distance_same_results_as_reference_implementation():

    D_reference = all_pairs_normalized_distances_reference(XY_incomplete)
    D_fast = all_pairs_normalized_distances(XY_incomplete)

    eq_(D_fast.shape, D_reference.shape)

    assert not np.isnan(D_reference).any(), "NaN in distance matrix"
    assert not np.isnan(D_fast).any(), "NaN in distance matrix"

    n_inf_reference = (~np.isfinite(D_reference)).sum()
    n_inf_fast = (~np.isfinite(D_fast)).sum()
    print("# infinity reference=%d fast=%d" % (
        n_inf_reference,
        n_inf_fast))
    eq_(n_inf_reference, n_inf_fast)
    diff = (D_fast - D_reference)
    finite_diffs = diff[np.isfinite(diff)]
    abs_diff = np.abs(finite_diffs)
    print(np.where(np.abs(diff) > 0.1))
    mae = np.mean(abs_diff)
    assert mae < 0.0001, \
        "Difference between distance matrices (MAE=%0.4f)" % mae


def test_normalized_distance_faster_than_reference_implementation():
    start_t = time()
    all_pairs_normalized_distances(XY_incomplete, verbose=False)
    fast_t = time() - start_t
    start_t = time()
    all_pairs_normalized_distances_reference(XY_incomplete)
    reference_t = time() - start_t
    print("Fast implementation: %0.2fs, reference implementation: %0.2fs" % (
        fast_t, reference_t))
    assert reference_t / fast_t > 2, "Expected 2x performance gain"
