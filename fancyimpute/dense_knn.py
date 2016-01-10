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

from six.moves import range
import numpy as np
import time

from .solver import Solver
from .normalized_distance import all_pairs_normalized_distances


def knn_initialize(X, missing_mask, verbose=False):
    """
    Fill X with NaN values if necessary, construct the n_samples x n_samples
    distance matrix and set the self-distance of each row to infinity.
    """
    X_row_major = X.copy("C")
    if missing_mask.sum() != np.isnan(X_row_major).sum():
        # if the missing values have already been zero-filled need
        # to put NaN's back in the data matrix for the distances function
        X_row_major[missing_mask] = np.nan
    D = all_pairs_normalized_distances(X_row_major, verbose=verbose)
    # set diagonal of distance matrix to infinity since we don't want
    # points considering themselves as neighbors
    for i in range(X.shape[0]):
        D[i, i] = np.inf
    return X_row_major, D


def knn_impute(X, missing_mask, k, verbose=False):
    """
    Fill in the given incomplete matrix using k-nearest neighbor imputation.

    This version is a simpler algorithm meant primarily for testing but
    surprisingly it's faster for many (but not all) dataset sizes, particularly
    when most of the columns are missing in any given row. The crucial
    bottleneck is the call to numpy.argpartition for every missing element
    in the array.

    Parameters
    ----------
    X : np.ndarray
        Matrix to fill of shape (n_samples, n_features)

    missing_mask : np.ndarray
        Boolean array of same shape as X

    k : int

    verbose : bool

    Modifies X by replacing its missing values with weighted averages of
    similar rows. Returns the modified X.
    """
    start_t = time.time()
    n_rows, n_cols = X.shape
    # put the missing mask in column major order since it's accessed
    # one column at a time
    missing_mask_column_major = np.asarray(missing_mask, order="F")
    X_row_major, D = knn_initialize(X, missing_mask, verbose=verbose)
    D_reciprocal = 1.0 / D
    neighbor_weights = np.zeros(k, dtype="float32")
    dot = np.dot
    for i in range(n_rows):
        missing_indices = np.where(missing_mask[i])[0]

        if verbose and i % 100 == 0:
            print(
                "[DenseKNN] Imputing row %d/%d with %d missing columns, elapsed time: %0.3f" % (
                    i + 1,
                    n_rows,
                    len(missing_indices),
                    time.time() - start_t))
        d = D[i, :]
        inv_d = D_reciprocal[i, :]
        for j in missing_indices:
            column = X[:, j]
            rows_missing_feature = missing_mask_column_major[:, j]
            d = d.copy()
            d[rows_missing_feature] = np.inf
            neighbor_indices = np.argpartition(d, k)[:k]
            neighbor_weights = inv_d[neighbor_indices]
            X[i, j] = (
                dot(column[neighbor_indices], neighbor_weights) /
                neighbor_weights.sum()
            )
    return X


def knn_impute_experimental(X, missing_mask, k, verbose=False):
    """
    Fill in the given incomplete matrix using k-nearest neighbor imputation.
    This version does a lot more work in a (misguided) attempt at cleverness.
    Has been observed to be a lot faster for 1/4 missing images matrix
    with 1000 rows and ~9000 columns.

    Parameters
    ----------
    X : np.ndarray
        Matrix to fill of shape (n_samples, n_features)

    missing_mask : np.ndarray
        Boolean array of same shape as X

    k : int

    verbose : bool

    Modifies X by replacing its missing values with weighted averages of
    similar rows. Returns the modified X.
    """
    start_t = time.time()
    n_rows, n_cols = X.shape
    X_row_major, D = knn_initialize(X, missing_mask, verbose=verbose)
    D_sorted_indices = np.argsort(D, axis=1)
    X_column_major = X_row_major.copy(order="F")

    dot = np.dot

    # preallocate array to prevent repeated creation in the following loops
    neighbor_weights = np.ones(k, dtype=X.dtype)

    missing_mask_column_major = np.asarray(missing_mask, order="F")
    observed_mask_column_major = ~missing_mask_column_major

    for i in range(n_rows):
        missing_columns = np.where(missing_mask[i])[0]
        if verbose and i % 100 == 0:
            print(
                "[DenseKNN] Imputing row %d/%d with %d missing columns, elapsed time: %0.3f" % (
                    i + 1,
                    n_rows,
                    len(missing_columns),
                    time.time() - start_t))
        n_missing_columns = len(missing_columns)
        if n_missing_columns == 0:
            continue

        row_distances = D[i, :]
        neighbor_indices = D_sorted_indices[i, :]
        X_missing_columns = X_column_major[:, missing_columns]

        # precompute these for the fast path where the k nearest neighbors
        # are not missing the feature value we're currently trying to impute
        k_nearest_indices = neighbor_indices[:k]
        np.divide(1.0, row_distances[k_nearest_indices], out=neighbor_weights)
        # optimistically impute all the columns from the k nearest neighbors
        # we'll have to back-track for some of the columns for which
        # one of the neighbors did not have a value
        X_knn = X_missing_columns[k_nearest_indices, :]
        weighted_average_of_neighboring_rows = dot(
            X_knn.T,
            neighbor_weights)
        sum_weights = neighbor_weights.sum()
        weighted_average_of_neighboring_rows /= sum_weights
        imputed_values = weighted_average_of_neighboring_rows

        observed_mask_missing_columns = observed_mask_column_major[:, missing_columns]
        observed_mask_missing_columns_sorted = observed_mask_missing_columns[
            neighbor_indices, :]

        # We can determine the maximum number of other rows that must be
        # inspected across all features missing for this row by
        # looking at the column-wise running sums of the observed feature
        # matrix.
        observed_cumulative_sum = observed_mask_missing_columns_sorted.cumsum(axis=0)
        sufficient_rows = (observed_cumulative_sum == k)
        n_rows_needed = sufficient_rows.argmax(axis=0) + 1
        max_rows_needed = n_rows_needed.max()

        if max_rows_needed == k:
            # if we never needed more than k rows then we're done after the
            # optimistic averaging above, so go on to the next sample
            X[i, missing_columns] = imputed_values
            continue

        # truncate all the sorted arrays to only include the necessary
        # number of rows (should significantly speed up the "slow" path)
        necessary_indices = neighbor_indices[:max_rows_needed]
        d_sorted = row_distances[necessary_indices]
        X_missing_columns_sorted = X_missing_columns[necessary_indices, :]
        observed_mask_missing_columns_sorted = observed_mask_missing_columns_sorted[
            :max_rows_needed, :]

        for missing_column_idx in range(n_missing_columns):
            # since all the arrays we're looking into have already been
            # sliced out at the missing features, we need to address these
            # features from 0..n_missing using missing_idx rather than j
            if n_rows_needed[missing_column_idx] == k:
                assert np.isfinite(imputed_values[missing_column_idx]), \
                    "Expected finite imputed value #%d (column #%d for row %d)" % (
                        missing_column_idx,
                        missing_columns[missing_column_idx],
                        i)
                continue
            row_mask = observed_mask_missing_columns_sorted[:, missing_column_idx]
            sorted_column_values = X_missing_columns_sorted[:, missing_column_idx]
            neighbor_values = sorted_column_values[row_mask][:k]
            neighbor_distances = d_sorted[row_mask][:k]
            np.divide(1.0, neighbor_distances, out=neighbor_weights)
            imputed_values[missing_column_idx] = (
                dot(neighbor_values, neighbor_weights) /
                neighbor_weights.sum()
            )
        X[i, missing_columns] = imputed_values
    return X


class DenseKNN(Solver):
    """
    k-Nearest Neighbors Imputation for arrays with missing data.
    Works only on dense arrays with a moderate number of rows.

    Assumes that each feature has been centered and rescaled to have
    mean 0 and variance 1.
    """
    def __init__(self, k=5, verbose=True, orientation="rows"):
        Solver.__init__(self)
        self.k = k
        self.verbose = verbose
        self.orientation = orientation

    def solve(self, X, missing_mask):
        if self.orientation == "columns":
            X = X.T
            missing_mask = missing_mask.T
        elif self.orientation != "rows":
            raise ValueError(
                "Orientation must be either 'rows' or 'columns', got: %s" % (
                    self.orientation,))
        X = knn_impute(
            X=X,
            missing_mask=missing_mask,
            k=self.k,
            verbose=self.verbose)
        if self.orientation == "columns":
            X = X.T
        n_missing_after_imputation = np.isnan(X).sum()
        assert n_missing_after_imputation == 0, \
            "Expected all values to be filled but got %d missing" % (
                n_missing_after_imputation,)
        return X
