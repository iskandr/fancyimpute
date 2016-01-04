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
            n_imputations=1,
            init_fill_method="random",
            min_difference_between_iters=0.001,
            min_fraction_improvement=0.999,
            patience=5,
            verbose=True):
        self.rank = rank
        self.max_iters = max_iters
        self.n_imputations = n_imputations
        self.init_fill_method = init_fill_method
        self.patience = patience
        self.min_fraction_improvement = min_fraction_improvement
        self.verbose = verbose

    def single_imputation(self, X):
        X_filled, missing_mask = self.prepare_data(
            X,
            inplace=False,
            fill_method=self.init_fill_method)
        observed_mask = ~missing_mask
        best_mae = np.inf
        best_solution = X_filled
        iters_since_best = 0
        for i in range(self.max_iters):
            tsvd = TruncatedSVD(self.rank)
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

    def multiple_imputations(self, X):
        return [self.single_imputation(X) for _ in range(self.n_imputations)]

    def complete(self, X):
        imputations = self.multiple_imputations(X)
        return np.mean(imputations, axis=0)
