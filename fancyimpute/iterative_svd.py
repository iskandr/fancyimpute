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

from sklearn.decomposition import TruncatedSVD
import numpy as np

from .solver import Solver
from .common import masked_mae


class IterativeSVD(Solver):
    def __init__(
            self,
            rank=10,
            convergence_threshold=0.00001,
            max_iters=200,
            gradual_rank_increase=True,
            svd_algorithm="arpack",
            init_fill_method="zero",
            min_value=None,
            max_value=None,
            verbose=True):
        Solver.__init__(
            self,
            fill_method=init_fill_method,
            min_value=min_value,
            max_value=max_value)
        self.rank = rank
        self.max_iters = max_iters
        self.svd_algorithm = svd_algorithm
        self.convergence_threshold = convergence_threshold
        self.gradual_rank_increase = gradual_rank_increase
        self.verbose = verbose

    def _converged(self, X_old, X_new, missing_mask):
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm_squared = (old_missing_values ** 2).sum()
        return (ssd / old_norm_squared) < self.convergence_threshold

    def solve(self, X, missing_mask):
        observed_mask = ~missing_mask
        X_filled = X
        for i in range(self.max_iters):
            # deviation from original svdImpute algorithm:
            # gradually increase the rank of our approximation
            if self.gradual_rank_increase:
                curr_rank = min(2 ** i, self.rank)
            else:
                curr_rank = self.rank
            tsvd = TruncatedSVD(curr_rank, algorithm=self.svd_algorithm)
            X_reduced = tsvd.fit_transform(X_filled)
            X_reconstructed = tsvd.inverse_transform(X_reduced)
            X_reconstructed = self.clip(X_reconstructed)
            mae = masked_mae(
                X_true=X,
                X_pred=X_reconstructed,
                mask=observed_mask)
            if self.verbose:
                print(
                    "[IterativeSVD] Iter %d: observed MAE=%0.6f" % (
                        i + 1, mae))
            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstructed,
                missing_mask=missing_mask)
            X_filled[missing_mask] = X_reconstructed[missing_mask]
            if converged:
                break
        return X_filled
