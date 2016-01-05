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

from sklearn.decomposition import TruncatedSVD
import numpy as np

from .solver import Solver
from .common import masked_mae


class IterativeSVD(Solver):
    def __init__(
            self,
            rank,
            max_iters=100,
            min_difference_between_iters=0.001,
            min_fraction_improvement=0.999,
            patience=5,
            svd_algorithm="arpack",
            init_fill_method="zero",
            n_imputations=1,
            normalize_columns=True,
            min_value=None,
            max_value=None,
            verbose=True):
        Solver.__init__(
            self,
            fill_method=init_fill_method,
            n_imputations=n_imputations,
            normalize_columns=normalize_columns,
            min_value=min_value,
            max_value=max_value)
        self.rank = rank
        self.max_iters = max_iters
        self.patience = patience
        self.svd_algorithm = svd_algorithm
        self.min_fraction_improvement = min_fraction_improvement
        self.verbose = verbose

    def solve(self, X, missing_mask):
        observed_mask = ~missing_mask
        best_mae = np.inf
        best_solution = X
        iters_since_best = 0
        X_filled = X
        tsvd = TruncatedSVD(self.rank, algorithm=self.svd_algorithm)
        for i in range(self.max_iters):
            X_reduced = tsvd.fit_transform(X_filled)
            X_reconstructed = tsvd.inverse_transform(X_reduced)
            mae = masked_mae(
                X_true=X,
                X_pred=X_reconstructed,
                mask=observed_mask)
            if self.verbose:
                print(
                    "[IterativeSVD] Iter %d: observed MAE=%0.6f" % (
                        i + 1, mae))
            X_filled = X.copy()
            X_filled[missing_mask] = X_reconstructed[missing_mask]
            if i == 0:
                best_mae = mae
                best_solution = X_filled.copy()
                iters_since_best = 0
            elif mae / best_mae < self.min_fraction_improvement:
                best_mae = mae
                best_solution = X_filled.copy()
                iters_since_best = 0
            elif iters_since_best > self.patience:
                if self.verbose:
                    print(
                        "[IterativeSVD] Patience exceeded on iter %d" % (
                            i + 1,))
                break
            else:
                iters_since_best += 1
        return best_solution
