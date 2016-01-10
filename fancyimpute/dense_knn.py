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
from .knn_helpers import all_pairs_normalized_distances


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
        start_t = time.time()
        if self.orientation == "columns":
            X = X.T
            missing_mask = missing_mask.T
        elif self.orientation != "rows":
            raise ValueError(
                "Orientation must be either 'rows' or 'columns', got: %s" % (
                    self.orientation,))
        X_with_nans = X.copy(order="C")
        X_with_nans[missing_mask] = np.nan
        X_with_nans_column_major = np.asarray(X_with_nans, order="F")

        D = all_pairs_normalized_distances(X_with_nans, verbose=self.verbose)
        D_sorted_indices = np.argsort(D, axis=1)
        n_rows = X.shape[0]
        missing_columns_for_each_row = [
            np.where(missing_mask[i])[0]
            for i in range(n_rows)
        ]
        k = self.k
        dot = np.dot

        # preallocate array to prevent repeated creation in the following loops
        neighbor_weights = np.ones(k, dtype=X.dtype)

        missing_mask_column_major = np.asarray(missing_mask, order="F")
        observed_mask_column_major = ~missing_mask_column_major

        for i in range(n_rows):
            missing_columns = missing_columns_for_each_row[i]
            if self.verbose and i % 100 == 0:
                print(
                    "[DenseKNN] Imputing row %d/%d with %d missing columns, elapsed time: %0.3f" % (
                        i + 1,
                        n_rows,
                        len(missing_columns),
                        time.time() - start_t))
            row_distances = D[i, :]
            missing_columns = missing_columns_for_each_row[i]
            n_missing_columns = len(missing_columns)
            neighbor_indices = D_sorted_indices[i, :]
            X_missing_columns = X_with_nans_column_major[:, missing_columns]

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
        if self.orientation == "columns":
            X = X.T
        n_missing_after_imputation = np.isnan(X).sum()
        assert n_missing_after_imputation == 0, \
            "Expected all values to be filled but got %d missing" % (n_missing_after_imputation,)
        return X
