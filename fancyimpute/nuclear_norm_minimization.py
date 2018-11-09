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

import cvxpy

from .solver import Solver

from sklearn.utils import check_array


class NuclearNormMinimization(Solver):
    """
    Simple implementation of "Exact Matrix Completion via Convex Optimization"
    by Emmanuel Candes and Benjamin Recht using cvxpy.
    """

    def __init__(
            self,
            require_symmetric_solution=False,
            min_value=None,
            max_value=None,
            error_tolerance=0.0001,
            max_iters=50000,
            verbose=True):
        """
        Parameters
        ----------
        require_symmetric_solution : bool
            Add symmetry constraint to convex problem

        min_value : float
            Smallest possible imputed value

        max_value : float
            Largest possible imputed value

        error_tolerance : bool
            Degree of error allowed on reconstructed values. If omitted then
            defaults to 0.0001

        max_iters : int
            Maximum number of iterations for the convex solver

        verbose : bool
            Print debug info
        """
        Solver.__init__(
            self,
            min_value=min_value,
            max_value=max_value)
        self.require_symmetric_solution = require_symmetric_solution
        self.error_tolerance = error_tolerance
        self.max_iters = max_iters
        self.verbose = verbose

    def _constraints(self, X, missing_mask, S, error_tolerance):
        """
        Parameters
        ----------
        X : np.array
            Data matrix with missing values filled in

        missing_mask : np.array
            Boolean array indicating where missing values were

        S : cvxpy.Variable
            Representation of solution variable
        """
        ok_mask = ~missing_mask
        masked_X = cvxpy.multiply(ok_mask, X)
        masked_S = cvxpy.multiply(ok_mask, S)
        abs_diff = cvxpy.abs(masked_S - masked_X)
        close_to_data = abs_diff <= error_tolerance
        constraints = [close_to_data]
        if self.require_symmetric_solution:
            constraints.append(S == S.T)

        if self.min_value is not None:
            constraints.append(S >= self.min_value)

        if self.max_value is not None:
            constraints.append(S <= self.max_value)

        return constraints

    def _create_objective(self, m, n):
        """
        Parameters
        ----------
        m, n : int
            Dimensions that of solution matrix
        Returns the objective function and a variable representing the
        solution to the convex optimization problem.
        """
        # S is the completed matrix
        shape = (m, n)
        S = cvxpy.Variable(shape, name="S")
        norm = cvxpy.norm(S, "nuc")
        objective = cvxpy.Minimize(norm)
        return S, objective

    def solve(self, X, missing_mask):
        X = check_array(X, force_all_finite=False)

        m, n = X.shape
        S, objective = self._create_objective(m, n)
        constraints = self._constraints(
            X=X,
            missing_mask=missing_mask,
            S=S,
            error_tolerance=self.error_tolerance)
        problem = cvxpy.Problem(objective, constraints)
        problem.solve(
            verbose=self.verbose,
            solver=cvxpy.SCS,
            max_iters=self.max_iters,
            # use_indirect, see: https://github.com/cvxgrp/cvxpy/issues/547
            use_indirect=False)
        return S.value
