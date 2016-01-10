# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import numpy as np


def all_pairs_normalized_distances(X, verbose=True):
    """
    We can't really compute distances over incomplete data since
    rows are missing different numbers of entries.
    The next best thing is the mean squared difference between two vectors
    (a normalized distance), which gets computed only over the columns that
    two vectors have in common. If two vectors have no features in common
    then their distance is infinity.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features) with missing entries
        marked using np.nan

    Returns a (n_samples, n_samples) matrix of pairwise normalized distances.
    """
    n_rows, n_cols = X.shape
    t_start = time.time()
    if verbose:
        print("Computing pairwise distances between %d samples" % (
            n_rows))
    # matrix of mean squared difference between between samples
    D = np.ones((n_rows, n_rows), dtype="float32", order="C") * np.inf

    # preallocate all the arrays that we would otherwise create in the
    # following loop and pass them as "out" parameters to NumPy ufuncs
    diffs = np.zeros_like(X)
    missing_differences = np.zeros_like(diffs, dtype=bool)
    missing_per_row = np.zeros(n_rows, dtype=int)
    observed_counts_per_row = np.zeros(n_rows, dtype=int)
    empty_rows = np.zeros(n_rows, dtype=bool)
    valid_rows = np.zeros_like(empty_rows)
    for i in range(n_rows):
        if verbose and i % 100 == 0:
            print("Computing distances for sample #%d/%d, elapsed time: %0.3f" % (
                i + 1,
                n_rows,
                time.time() - t_start))
        x = X[i, :]
        np.subtract(X, x.reshape((1, n_cols)), out=diffs)
        np.isnan(diffs, out=missing_differences)
        missing_differences.sum(axis=1, out=missing_per_row)
        np.equal(missing_per_row, n_cols, out=empty_rows)
        n_missing_rows = empty_rows.sum()
        if n_missing_rows == n_rows:
            print("No samples have sufficient overlap with sample %d" % (
                i,))
            continue

        # zero out all NaN's
        diffs[missing_differences] = 0
        # square each difference
        diffs **= 2
        # add up all the non-missing squared differences
        ssd = np.sum(diffs, axis=1)

        if n_missing_rows == 0:
            np.subtract(
                n_cols,
                missing_per_row,
                out=observed_counts_per_row)
            ssd /= observed_counts_per_row
            D[i, :] = ssd
        else:
            np.logical_not(empty_rows, out=valid_rows)
            diffs_slice = diffs[valid_rows, :]
            D[i, valid_rows] = np.nanmean(diffs_slice, axis=1)
        # set the distance between a sample and itself to infinity
        # since we're always going to be using these distances to find
        # rows which have features that the current row is missing
        D[i, i] = np.inf
    return D
