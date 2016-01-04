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
from sklearn.decomposition import randomized_svd


from .common import masked_mae, choose_solution_using_percentiles
from .solver import Solver


class SoftImpute(Solver):
    """
    Implementation of the SoftImpute algorithm from:
    "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
    by Mazumder, Hastie, and Tibshirani.


    Basic algorithm sketch:
    1. Initialize Z_old = 0.
    2. Do for λ1 > λ2 >...>λK:
    (a) Repeat:
        i. Compute Z_new ← S_λi (Observed(X)+Missing(Z_old))
        ii. If norm(Z_old - Z_new) / norm(Z_old) < eps: exit
        iii. Assign Z_old = Z_new
    (b) Assign Z[λi] = Z_new
    3. Output the sequence of solutions Z[λi]
    """
    def __init__(
            self,
            shrinkage_values=[40, 20, 10, 5, 1, 0.5, 0.1, 0.05, 0.01],
            convergence_threshold=0.001,
            max_iters=100,
            rank=None,
            n_power_iterations=5,
            init_fill_method="zero",
            verbose=True):
        self.shrinkage_values = shrinkage_values
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        self.rank = rank
        self.n_power_iterations = n_power_iterations
        self.init_fill_method = init_fill_method
        self.verbose = verbose

    def _converged(self, X_old, X_new, missing_mask):
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm = (old_missing_values ** 2).sum()
        return (ssd / old_norm) < self.convergence_threshold

    def _svd_step(self, X, shrinkage_value):
        if self.rank:
            # if we have a max rank then perform the faster randomized SVD
            (U, s, V) = randomized_svd(
                X,
                self.rank,
                n_iter=self.n_power_iterations)
        else:
            # perform a full rank SVD using ARPACK
            (U, s, V) = np.linalg.svd(
                X,
                full_matrices=False,
                compute_uv=True)
        s_sparse = np.maximum(s - shrinkage_value, 0)
        if self.verbose:
            print(
                "-- sparsity = %d/%d" % (
                    (s_sparse > 0).sum(),
                    len(s_sparse)))
        return np.dot(U, np.dot(np.diag(s_sparse), V))

    def _single_imputation(self, X_init, missing_mask, shrinkage_value):
        X_filled = X_init.copy()
        for i in range(self.max_iters):
            X_reconstruction = self._svd_step(X_filled, shrinkage_value)

            # print error on observed data
            if self.verbose:
                mae = masked_mae(
                    X_true=X_init,
                    X_pred=X_reconstruction,
                    mask=~missing_mask)
                print(
                    "[SoftImpute] Iter %d: observed MAE=%0.6f" % (
                        i + 1,
                        mae))

            if self._converged(
                    X_old=X_filled,
                    X_new=X_reconstruction,
                    missing_mask=missing_mask):
                break
            X_filled[missing_mask] = X_reconstruction[missing_mask]
        if self.verbose:
            print("[SoftImpute] Stopped after iteration %d for lambda=%f" % (
                i + 1,
                shrinkage_value))
        return X_filled

    def multiple_imputations(self, X):
        X_filled, missing_mask = self.prepare_data(
            X,
            inplace=False,
            fill_method=self.init_fill_method)

        results = []
        for shrinkage_value in reversed(sorted(self.shrinkage_values)):
            # traverse shrinkage values in decrease order
            # and use last solution as start for the next one
            X_result = self._single_imputation(
                X_filled,
                missing_mask,
                shrinkage_value)
            results.append(X_result)
        return results

    def complete(self, X):
        solutions = self.multiple_imputations(X)
        if len(solutions) == 1:
            return solutions[0]
        else:
            return choose_solution_using_percentiles(
                X,
                solutions,
                parameters=self.shrinkage_values,
                verbose=self.verbose)
