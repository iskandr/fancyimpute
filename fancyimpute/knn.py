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
            verbose=True,
            orientation="rows",
            use_argpartition=False,
            print_interval=100):
        """
        Parameters
        ----------
        k : int
            Number of neighboring rows to use for imputation.

        verbose : bool

        orientation : str

        use_argpartition : bool
           Use a more naive implementation of kNN imputation whichs calls
           numpy.argpartition for each row/column pair. May give NaN if fewer
           than k neighbors are available for a missing value.

        print_interval : int
        """

        Solver.__init__(self)
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
        X = self._impute_fn(
            X=X,
            missing_mask=missing_mask,
            k=self.k,
            verbose=self.verbose,
            print_interval=self.print_interval)
        if self.orientation == "columns":
            X = X.T
        n_missing_after_imputation = np.isnan(X).sum()
        assert n_missing_after_imputation == 0, \
            "Expected all values to be filled but got %d/%d missing" % (
                n_missing_after_imputation,
                X.shape[0] * X.shape[1])
        return X
