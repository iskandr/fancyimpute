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

from __future__ import absolute_import, print_function, division
import time

from six.moves import range
import numpy as np

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


def knn_impute_reference(
        X,
        missing_mask,
        k,
        verbose=False,
        print_interval=100):
    """
    Reference implementation of kNN imputation logic.
    """
    n_rows, n_cols = X.shape
    X_result, D = knn_initialize(X, missing_mask, verbose=verbose)

    # get rid of infinities, replace them with a very large number
    finite_distance_distance_mask = np.isfinite(D)
    effective_infinity = 10 ** 6 * D[finite_distance_distance_mask].max()
    D[~finite_distance_distance_mask] = effective_infinity

    for i in range(n_rows):
        for j in np.where(missing_mask[i, :])[0]:
            distances = D[i, :].copy()

            # any rows that don't have the value we're currently trying
            # to impute are set to infinite distances
            distances[missing_mask[:, j]] = effective_infinity
            neighbor_indices = np.argsort(distances)
            neighbor_distances = distances[neighbor_indices]

            # get rid of any infinite distance neighbors in the top k
            valid_distances = neighbor_distances < effective_infinity
            neighbor_distances = neighbor_distances[valid_distances][:k]
            neighbor_indices = neighbor_indices[valid_distances][:k]

            weights = 1.0 / neighbor_distances
            weight_sum = weights.sum()

            if weight_sum > 0:
                column = X[:, j]
                values = column[neighbor_indices]
                X_result[i, j] = np.dot(values, weights) / weight_sum
    return X_result


def knn_impute_with_argpartition(
        X,
        missing_mask,
        k,
        verbose=False,
        print_interval=100):
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

    Returns a row-major copy of X with imputed values.
    """
    start_t = time.time()
    n_rows, n_cols = X.shape
    # put the missing mask in column major order since it's accessed
    # one column at a time
    missing_mask_column_major = np.asarray(missing_mask, order="F")
    X_row_major, D = knn_initialize(X, missing_mask, verbose=verbose)
    # D[~np.isfinite(D)] = very_large_value
    D_reciprocal = 1.0 / D
    neighbor_weights = np.zeros(k, dtype="float32")
    dot = np.dot
    for i in range(n_rows):
        missing_indices = np.where(missing_mask[i])[0]

        if verbose and i % print_interval == 0:
            print(
                "Imputing row %d/%d with %d missing columns, elapsed time: %0.3f" % (
                    i + 1,
                    n_rows,
                    len(missing_indices),
                    time.time() - start_t))
        d = D[i, :]
        inv_d = D_reciprocal[i, :]
        for j in missing_indices:
            column = X[:, j]
            rows_missing_feature = missing_mask_column_major[:, j]
            d_copy = d.copy()
            # d_copy[rows_missing_feature] = very_large_value
            d_copy[rows_missing_feature] = np.inf
            neighbor_indices = np.argpartition(d_copy, k)[:k]
            if len(neighbor_indices) > 0:
                neighbor_weights = inv_d[neighbor_indices]
                X_row_major[i, j] = (
                    dot(column[neighbor_indices], neighbor_weights) /
                    neighbor_weights.sum()
                )
    return X_row_major


def knn_impute_optimistic(
        X,
        missing_mask,
        k,
        verbose=False, print_interval=100):
    """
    Fill in the given incomplete matrix using k-nearest neighbor imputation.

    This version assumes that most of the time the same neighbors will be
    used so first performs the weighted average of a row's k-nearest neighbors
    and checks afterward whether it was valid (due to possible missing values).

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
        if verbose and i % print_interval == 0:
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
            neighbor_distances = d_sorted[row_mask][:k]

            # may not have enough values in a column for all k neighbors
            k_or_less = len(neighbor_distances)
            usable_weights = neighbor_weights[:k_or_less]
            np.divide(
                1.0,
                neighbor_distances, out=usable_weights)
            neighbor_values = sorted_column_values[row_mask][:k_or_less]

            imputed_values[missing_column_idx] = (
                dot(neighbor_values, usable_weights) / usable_weights.sum())

        X[i, missing_columns] = imputed_values
    return X


def knn_impute_few_observed(
        X, missing_mask, k, verbose=False, print_interval=100):
    """
    Seems to be the fastest kNN implementation. Pre-sorts each rows neighbors
    and then filters these sorted indices using each columns mask of
    observed values.

    Important detail: If k observed values are not available then uses fewer
    than k neighboring rows.

    Parameters
    ----------
    X : np.ndarray
        Matrix to fill of shape (n_samples, n_features)

    missing_mask : np.ndarray
        Boolean array of same shape as X

    k : int

    verbose : bool
    """
    start_t = time.time()
    n_rows, n_cols = X.shape
    # put the missing mask in column major order since it's accessed
    # one column at a time
    missing_mask_column_major = np.asarray(missing_mask, order="F")
    observed_mask_column_major = ~missing_mask_column_major
    X_column_major = X.copy(order="F")
    X_row_major, D = knn_initialize(X, missing_mask, verbose=verbose)
    # get rid of infinities, replace them with a very large number
    finite_distance_distance_mask = np.isfinite(D)
    effective_infinity = 10 ** 6 * D[finite_distance_distance_mask].max()
    D[~finite_distance_distance_mask] = effective_infinity
    D_sorted = np.argsort(D, axis=1)
    inv_D = 1.0 / D
    D_valid_mask = D < effective_infinity
    valid_distances_per_row = D_valid_mask.sum(axis=1)

    # trim the number of other rows we consider to exclude those
    # with infinite distances
    D_sorted = [
        D_sorted[i, :count]
        for i, count in enumerate(valid_distances_per_row)
    ]

    dot = np.dot
    for i in range(n_rows):
        missing_row = missing_mask[i, :]
        missing_indices = np.where(missing_row)[0]
        row_weights = inv_D[i, :]
        # row_sorted_indices = D_sorted_indices[i]
        if verbose and i % print_interval == 0:
            print(
                "Imputing row %d/%d with %d missing columns, elapsed time: %0.3f" % (
                    i + 1,
                    n_rows,
                    len(missing_indices),
                    time.time() - start_t))
        # row_neighbor_indices = neighbor_indices[i]
        candidate_neighbor_indices = D_sorted[i]

        for j in missing_indices:
            observed = observed_mask_column_major[:, j]
            sorted_observed = observed[candidate_neighbor_indices]
            observed_neighbor_indices = candidate_neighbor_indices[sorted_observed]
            k_nearest_indices = observed_neighbor_indices[:k]
            weights = row_weights[k_nearest_indices]
            weight_sum = weights.sum()
            if weight_sum > 0:
                column = X_column_major[:, j]
                values = column[k_nearest_indices]
                X_row_major[i, j] = dot(values, weights) / weight_sum
    return X_row_major
