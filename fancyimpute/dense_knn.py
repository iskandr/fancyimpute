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

from .solver import Solver


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

    def _compute_distances(self, X):
        n_samples, n_cols = X.shape

        if self.verbose:
            print("[DenseKNN] Computing pairwise distances between %d samples" % (
                n_samples))
        # matrix of mean squared difference between between samples
        D = np.zeros((n_samples, n_samples), dtype="float32")
        for i in range(n_samples):
            x = X[i, :]
            diffs = X - x.reshape((1, n_cols))
            observed = np.isfinite(diffs)
            observed_counts_per_row = observed.sum(axis=1)
            valid_rows = observed_counts_per_row > 0
            D[i, ~valid_rows] = np.inf
            if valid_rows.sum() == 0:
                print("No samples have sufficient overlap with sample %d" % (
                    i,))
            else:
                D[i, valid_rows] = np.nanmean(
                    diffs[valid_rows, :] ** 2,
                    axis=1)
        return D

    def solve(self, X, missing_mask):
        if self.orientation == "columns":
            X = np.asarray(X.T, order="F")
            missing_mask = np.asarray(missing_mask.T, order="F")
        elif self.orientation != "rows":
            raise ValueError(
                "Orientation must be either 'rows' or 'columns', got: %s" % (
                    self.orientation,))
        X_with_nans = X.copy()
        X_with_nans[missing_mask] = np.nan
        D = self._compute_distances(X_with_nans)
        n_rows = X.shape[0]
        missing_indices = [
            np.where(missing_mask[i])[0]
            for i in range(n_rows)
        ]
        for i in range(n_rows):
            if self.verbose and i % 100 == 0:
                print(
                    "[DenseKNN] Imputing row %d/%d with %d missing columns" % (
                        i,
                        n_rows,
                        len(missing_indices[i]),))
            d = D[i, :]
            for j in missing_indices[i]:
                column = X[:, j]
                rows_missing_feature = missing_mask[:, j]
                d_valid = d.copy()
                d_valid[rows_missing_feature] = np.inf
                sorted_indices = np.argsort(d_valid)
                neighbor_indices = sorted_indices[:self.k]
                neighbor_dists = d[neighbor_indices]
                # make sure no infininities snuck in
                sane_dist = np.isfinite(neighbor_dists)
                neighbor_indices = neighbor_indices[sane_dist]
                neighbor_dists = neighbor_dists[sane_dist]
                neighbor_weights = 1.0 / neighbor_dists
                X[i, j] = (
                    (column[neighbor_indices] * neighbor_weights).sum() /
                    neighbor_weights.sum()
                )
        if self.orientation == "columns":
            X = X.T
        return X
