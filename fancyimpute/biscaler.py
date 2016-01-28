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

from six.moves import range

import numpy as np


class BiScaler(object):
    """
    Iterative estimation of row and column centering/scaling
    using the algorithm from page 31 of:
        Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares
    """

    def __init__(
            self,
            center_rows=True,
            center_columns=True,
            scale_rows=True,
            scale_columns=True,
            min_value=None,
            max_value=None,
            max_iters=100,
            tolerance=0.001,
            verbose=True):
        self.center_rows = center_rows
        self.center_columns = center_columns
        self.scale_rows = scale_rows
        self.scale_columns = scale_columns
        self.min_value = min_value
        self.max_value = max_value
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.verbose = verbose

    def estimate_row_means(
            self,
            X,
            observed,
            column_means,
            column_scales):
        """
        row_center[i] =
        sum{j in observed[i, :]}{
            (1 / column_scale[j]) * (X[i, j] - column_center[j])
        }
        ------------------------------------------------------------
        sum{j in observed[i, :]}{1 / column_scale[j]}
        """

        n_rows, n_cols = X.shape

        column_means = np.asarray(column_means)
        if len(column_means) != n_cols:
            raise ValueError("Expected length %d but got shape %s" % (
                n_cols, column_means.shape))
        X = X - column_means.reshape((1, n_cols))
        column_weights = 1.0 / column_scales
        X *= column_weights.reshape((1, n_cols))
        row_means = np.zeros(n_rows, dtype=X.dtype)
        row_residual_sums = np.nansum(X, axis=1)
        for i in range(n_rows):
            row_mask = observed[i, :]
            sum_weights = column_weights[row_mask].sum()
            row_means[i] = row_residual_sums[i] / sum_weights
        return row_means

    def estimate_column_means(
            self,
            X,
            observed,
            row_means,
            row_scales):
        """
        column_center[j] =
        sum{i in observed[:, j]}{
            (1 / row_scale[i]) * (X[i, j]) - row_center[i])
        }
        ------------------------------------------------------------
        sum{i in observed[:, j]}{1 / row_scale[i]}
        """
        n_rows, n_cols = X.shape
        row_means = np.asarray(row_means)

        if len(row_means) != n_rows:
            raise ValueError("Expected length %d but got shape %s" % (
                n_rows, row_means.shape))
        column_means = np.zeros(n_cols, dtype=X.dtype)

        X = X - row_means.reshape((n_rows, 1))
        row_weights = 1.0 / row_scales
        X *= row_weights.reshape((n_rows, 1))
        col_residual_sums = np.nansum(X, axis=0)
        for j in range(n_cols):
            col_mask = observed[:, j]
            sum_weights = row_weights[col_mask].sum()
            column_means[j] = col_residual_sums[j] / sum_weights
        return column_means

    def center(self, X, row_means, column_means, inplace=False):
        n_rows, n_cols = X.shape
        row_means = np.asarray(row_means)
        column_means = np.asarray(column_means)
        if len(row_means) != n_rows:
            raise ValueError("Expected length %d but got shape %s" % (
                n_rows, row_means.shape))
        if len(column_means) != n_cols:
            raise ValueError("Expected length %d but got shape %s" % (
                n_cols, column_means.shape))
        if not inplace:
            X = X.copy()
        X -= row_means.reshape((n_rows, 1))
        X -= column_means.reshape((1, n_cols))
        return X

    def rescale(self, X, row_scales, column_scales, inplace=False):
        if not inplace:
            X = X.copy()
        n_rows, n_cols = X.shape
        X /= row_scales.reshape((n_rows, 1))
        X /= column_scales.reshape((1, n_cols))
        return X

    def estimate_row_scales(
            self,
            X_centered,
            column_scales):
        """
        row_scale[i]**2 =
        mean{j in observed[i, :]}{
            (X[i, j] - row_center[i] - column_center[j]) ** 2
            --------------------------------------------------
                        column_scale[j] ** 2
        }
        """
        n_rows, n_cols = X_centered.shape
        column_scales = np.asarray(column_scales)
        if len(column_scales) != n_cols:
            raise ValueError("Expected length %d but got shape %s" % (
                n_cols, column_scales))
        row_variances = np.nanmean(
            X_centered ** 2 / (column_scales ** 2).reshape((1, n_cols)),
            axis=1)
        row_variances[row_variances == 0] = 1.0
        assert len(row_variances) == n_rows, "%d != %d" % (
            len(row_variances),
            n_rows)
        return np.sqrt(row_variances)

    def estimate_column_scales(
            self,
            X_centered,
            row_scales):
        """
        column_scale[j] ** 2 =
          mean{i in observed[:, j]}{
            (X[i, j] - row_center[i] - column_center[j]) ** 2
            -------------------------------------------------
                        row_scale[i] ** 2
        }
        """
        n_rows, n_cols = X_centered.shape
        row_scales = np.asarray(row_scales)

        if len(row_scales) != n_rows:
            raise ValueError("Expected length %s, got shape %s" % (
                n_rows, row_scales.shape,))

        column_variances = np.nanmean(
            X_centered ** 2 / (row_scales ** 2).reshape((n_rows, 1)),
            axis=0)
        column_variances[column_variances == 0] = 1.0
        assert len(column_variances) == n_cols, "%d != %d" % (
            len(column_variances),
            n_cols)
        return np.sqrt(column_variances)

    def residual(self, X_normalized):
        total = 0
        if self.center_rows:
            row_means = np.nanmean(X_normalized, axis=1)
            total += (row_means ** 2).sum()

        if self.center_columns:
            column_means = np.nanmean(X_normalized, axis=0)
            total += (column_means ** 2).sum()

        if self.scale_rows:
            row_variances = np.nanvar(X_normalized, axis=1)
            row_variances[row_variances == 0] = 1.0
            total += (np.log(row_variances) ** 2).sum()

        if self.scale_columns:
            column_variances = np.nanvar(X_normalized, axis=0)
            column_variances[column_variances == 0] = 1.0
            total += (np.log(column_variances) ** 2).sum()

        return total

    def clamp(self, X, inplace=False):
        if not inplace:
            X = X.copy()
        if self.min_value is not None:
            X[X < self.min_value] = self.min_value
        if self.max_value is not None:
            X[X > self.max_value] = self.max_value
        return X

    def fit(self, X):
        X = self.clamp(X)
        n_rows, n_cols = X.shape
        dtype = X.dtype

        # To avoid inefficient memory access we keep around two copies
        # of the array, one contiguous in the rows and the other
        # contiguous in the columns
        X_row_major = np.asarray(X, order="C")
        X_column_major = np.asarray(X, order="F")

        observed_row_major = ~np.isnan(X_row_major)
        n_observed_per_row = observed_row_major.sum(axis=1)
        n_empty_rows = (n_observed_per_row == 0).sum()

        if n_empty_rows > 0:
            raise ValueError("%d rows have no observed values" % n_empty_rows)

        observed_column_major = np.asarray(observed_row_major, order="F")
        n_observed_per_column = observed_column_major.sum(axis=0)
        n_empty_columns = (n_observed_per_column == 0).sum()
        if n_empty_columns > 0:
            raise ValueError("%d columns have no observed values" % (
                n_empty_columns,))
        # initialize by assuming that rows are zero-mean/unit variance and
        # with a direct estimate of mean and standard deviation
        # of each column
        row_means = np.zeros(n_rows, dtype=dtype)
        row_scales = np.ones(n_rows, dtype=dtype)

        if self.center_columns:
            column_means = np.nanmean(X, axis=0)
        else:
            column_means = np.zeros(n_cols, dtype=dtype)

        if self.scale_columns:
            column_scales = np.nanstd(X, axis=0)
            column_scales[column_scales == 0] = 1.0
        else:
            column_scales = np.ones(n_cols, dtype=dtype)

        last_residual = self.residual(X)
        if self.verbose:
            print("[BiScaler] Initial log residual value = %f" % (
                np.log(last_residual),))
        for i in range(self.max_iters):
            if last_residual == 0:
                # already have a perfect fit, so let's get out of here
                print("[BiScaler] No room for improvement")
                break

            assert len(column_means) == n_cols, \
                "Wrong number of column means, expected %d but got %d" % (
                    n_cols,
                    len(column_means))
            assert len(column_scales) == n_cols, \
                "Wrong number of column scales, expected %d but got %d" % (
                    n_cols,
                    len(column_scales))
            assert len(row_means) == n_rows, \
                "Wrong number of row means, expected %d but got %d" % (
                    n_rows,
                    len(row_means))
            assert len(row_scales) == n_rows, \
                "Wrong number of row scales, expected %d but got %d" % (
                    n_rows,
                    len(row_scales))

            if self.center_rows:
                row_means = self.estimate_row_means(
                    X=X_row_major,
                    observed=observed_row_major,
                    column_means=column_means,
                    column_scales=column_scales)
            if self.center_columns:
                column_means = self.estimate_column_means(
                    X=X_column_major,
                    observed=observed_column_major,
                    row_means=row_means,
                    row_scales=row_scales)

            X_centered = self.center(
                X,
                row_means,
                column_means)
            if self.scale_rows:
                row_scales = self.estimate_row_scales(
                    X_centered=X_centered,
                    column_scales=column_scales)
            if self.scale_columns:
                column_scales = self.estimate_column_scales(
                    X_centered=X_centered,
                    row_scales=row_scales)

            X_normalized = self.rescale(X_centered, row_scales, column_scales)
            residual = self.residual(X_normalized)
            change_in_residual = last_residual - residual
            if self.verbose:
                print("[BiScaler] Iter %d: log residual = %f, log improvement ratio=%f" % (
                    i + 1,
                    np.log(residual),
                    np.log(last_residual / residual)))
            if change_in_residual / last_residual < self.tolerance:
                break
            last_residual = residual
        self.row_means = row_means
        self.row_scales = row_scales
        self.column_means = column_means
        self.column_scales = column_scales

    def transform(self, X):
        X = np.asarray(X).copy()
        X = self.center(X, self.row_means, self.column_means, inplace=True)
        X = self.rescale(X, self.row_scales, self.column_scales, inplace=True)
        return X

    def inverse_transform(self, X, inplace=False):
        X = np.asarray(X)
        if not inplace:
            X = X.copy()
        X = self.rescale(
            X,
            1.0 / self.row_scales,
            1.0 / self.column_scales,
            inplace=True)
        X = self.center(X, -self.row_means, -self.column_means, inplace=True)
        return self.clamp(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
