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

from .solver import Solver


class SimpleFill(Solver):
    def __init__(self, fill_method="mean", min_value=None, max_value=None):
        """
        Possible values for fill_method:
            "zero": fill missing entries with zeros
            "mean": fill with column means
            "median" : fill with column medians
            "min": fill with min value per column
            "random": fill with gaussian noise according to mean/std of column
        """
        Solver.__init__(
            self,
            fill_method=fill_method,
            min_value=None,
            max_value=None)
        self.filler = None

    def solve(self, X, missing_mask):
        """
        Since X is given to us already filled, just return it.
        """
        return X

    def fit(self, X, missing_mask):
        """Just for compatibility"""
        fill_method = self.fill_method
        if fill_method not in ("zero", "mean", "median", "min", "random"):
            raise ValueError("Invalid fill method: '%s'" % (fill_method))
        elif fill_method == "zero":
            # replace NaN's with 0
            X[missing_mask] = 0
            self.filler = 0
        elif fill_method == "mean":
            from numpy import nanmean
            self.filler = nanmean(X, axis=0)
            X = self.fill_columns_with_value(X, missing_mask, self.filler)
        elif fill_method == "median":
            from numpy import nanmedian
            self.filler = nanmedian(X, axis=0)
            X = self.fill_columns_with_value(X, missing_mask, self.filler)
        elif fill_method == "min":
            from numpy import nanmin
            self.filler = nanmin(X, axis=0)
            X = self.fill_columns_with_value(X, missing_mask, self.filler)
        elif fill_method == "random":
            from numpy import nanmean, nanstd
            mean_value = nanmean(X, axis=0)
            std_value = nanstd(X, axis=0)
            self.filler = [mean_value, std_value]

            X = self.fill_columns_with_random(X, missing_mask, self.filler)
        return X

    def transform(self, X, missing_mask):
        """Just for compatibility"""
        fill_method = self.fill_method
        if fill_method not in ("zero", "mean", "median", "min", "random"):
            raise ValueError("Invalid fill method: '%s'" % (fill_method))
        elif fill_method == "zero":
            # replace NaN's with 0
            X[missing_mask] = 0
        elif fill_method == "mean":
            X = self.fill_columns_with_value(X, missing_mask, self.filler)
        elif fill_method == "median":
            X = self.fill_columns_with_value(X, missing_mask, self.filler)
        elif fill_method == "min":
            X = self.fill_columns_with_value(X, missing_mask, self.filler)
        elif fill_method == "random":
            X = self.fill_columns_with_random(X, missing_mask, self.filler)
        return X

    @staticmethod
    def fill_columns_with_value(X, missing_mask, fill_values):
        from numpy import all, isnan
        assert len(fill_values) == X.shape[1]
        for col_idx, filler in enumerate(fill_values):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            if all(isnan(filler)):
                filler = 0
            X[missing_col, col_idx] = filler
        return X

    @staticmethod
    def fill_columns_with_random(X, missing_mask, fill_values):
        from numpy import random, array, isclose
        mean_values = fill_values[0]
        std_values = fill_values[1]
        assert len(mean_values) == X.shape[1]
        for col_idx in range(len(mean_values)):
            mean = mean_values[col_idx]
            std = std_values[col_idx]
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            if isclose(std, 0):
                X[missing_col, col_idx] = array([mean] * n_missing)
            else:
                X[missing_col, col_idx] = random.randn(n_missing) * std + mean
        return X

