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
import numpy as np

from .solver import Solver
from .knn_helpers import knn_impute_few_observed, knn_impute_with_argpartition


class KNN(Solver):
    """
    k-Nearest Neighbors imputation for arrays with missing data.
    Works only on dense arrays with at most a few thousand rows.

    Assumes that each feature has been centered and rescaled to have
    mean 0 and variance 1.
    """
    def __init__(
            self,
            k=5,
            orientation="rows",
            use_argpartition=False,
            print_interval=100,
            min_value=None,
            max_value=None,
            normalizer=None,
            verbose=True):
        """
        Parameters
        ----------
        k : int
            Number of neighboring rows to use for imputation.

        orientation : str
            Which axis of the input matrix should be treated as a sample
            (default is "rows" but can also be "columns")

        use_argpartition : bool
           Use a more naive implementation of kNN imputation whichs calls
           numpy.argpartition for each row/column pair. May give NaN if fewer
           than k neighbors are available for a missing value.

        print_interval : int

        min_value : float
            Minimum possible imputed value

        max_value : float
            Maximum possible imputed value

        normalizer : object
            Any object (such as BiScaler) with fit() and transform() methods

        verbose : bool
        """
        Solver.__init__(
            self,
            min_value=min_value,
            max_value=max_value,
            normalizer=normalizer)
        self.k = k
        self.verbose = verbose
        self.orientation = orientation
        self.print_interval = print_interval
        if use_argpartition:
            self._impute_fn = knn_impute_with_argpartition
        else:
            self._impute_fn = knn_impute_few_observed

    def solve(self, X, missing_mask):
        if self.orientation == "columns":
            X = X.T
            missing_mask = missing_mask.T

        elif self.orientation != "rows":
            raise ValueError(
                "Orientation must be either 'rows' or 'columns', got: %s" % (
                    self.orientation,))

        X_imputed = self._impute_fn(
            X=X,
            missing_mask=missing_mask,
            k=self.k,
            verbose=self.verbose,
            print_interval=self.print_interval)

        failed_to_impute = np.isnan(X_imputed)
        n_missing_after_imputation = failed_to_impute.sum()
        if n_missing_after_imputation != 0:
            print("[KNN] Warning: %d/%d still missing after imputation, replacing with 0" % (
                n_missing_after_imputation,
                X.shape[0] * X.shape[1]))
            X_imputed[failed_to_impute] = X[failed_to_impute]

        if self.orientation == "columns":
            X_imputed = X_imputed.T

        return X_imputed
