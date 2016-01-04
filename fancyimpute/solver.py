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

import numpy as np

from .common import generate_random_column_samples


class Solver(object):

    def _check_input(self, X):
        if len(X.shape) != 2:
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))

    def _check_missing_value_mask(self, missing):
        if not missing.any():
            raise ValueError("Input matrix is not missing any values")
        if missing.all():
            raise ValueError("Input matrix must have some non-missing values")

    def _fill_columns_with_fn(self, X, missing_mask, col_fn):
        for col_idx in range(X.shape[1]):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            col_data = X[:, col_idx]
            fill_values = col_fn(col_data)
            X[missing_col, col_idx] = fill_values

    def fill(self, X, fill_method, inplace=False):
        """
        Parameters
        ----------
        X : np.array
            Data array containing NaN entries

        missing_mask : np.array
            Boolean array indicating where NaN entries are

        fill_method : str
            "zero": fill missing entries with zeros
            "mean": fill with column means
            "median" : fill with column medians
            "min": fill with min value per column
            "random": fill with gaussian noise according to mean/std of column

        inplace : bool
            Modify matrix or fill a copy
        """
        missing_mask = np.isnan(X)
        self._check_missing_value_mask(missing_mask)

        if not inplace:
            X = X.copy()

        if fill_method not in ("zero", "mean", "median", "min", "random"):
            raise ValueError("Invalid fill method: '%s'" % (fill_method))
        elif fill_method == "zero":
            # replace NaN's with 0
            X[missing_mask] = 0
        elif fill_method == "mean":
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif fill_method == "median":
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        elif fill_method == "min":
            self._fill_columns_with_fn(X, missing_mask, np.nanmin)
        elif fill_method == "random":
            self._fill_columns_with_fn(
                X,
                missing_mask,
                col_fn=generate_random_column_samples)
        return X, missing_mask

    def prepare_data(self, X, fill_method="zero", inplace=False):
        """
        Check to make sure that the input matrix and its mask of missing
        values are valid, then fill the missing entries of X according to
        the method specified by `fill_method`:
            "zero": fill missing entries with zeros
            "mean": fill with column means
            "median" : fill with column medians
            "min": fill with min value per column
            "random": fill with gaussian noise according to mean/std of column

        Returns initialized matrix X containing no NaN values and a mask of
        where the values previously were.
        """
        X = np.asarray(X)
        self._check_input(X)
        return self.fill(X, fill_method=fill_method, inplace=inplace)

    def clip_result(self, X, min_value=None, max_value=None):
        X = np.asarray(X)
        if min_value is not None:
            X[X < min_value] = min_value
        if max_value is not None:
            X[X > max_value] = max_value
        return X

    def single_imputation(self, X):
        raise ValueError("%s.single_imputation not yet implemented!" % (
            self.__class__.__name__,))

    def multiple_imputations(self, X):
        raise ValueError("%s.multiple_imputations not yet implemented!" % (
            self.__class__.__name__,))

    def complete(self, X):
        raise ValueError("%s.complete not yet implemented!" % (
            self.__class__.__name__,))
