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

import cvxpy
import numpy as np


class ConvexSolver(object):
    """
    Simple implementation of "Exact Matrix Completion via Convex Optimization"
    by Emmanuel Candes and Benjamin Recht using cvxpy.
    """

    def __init__(
            self,
            require_symmetric_solution=False,
            min_value=None,
            max_value=None,
            error_tolerance=0.0):
        self.require_symmetric_solution = require_symmetric_solution
        self.min_value = min_value
        self.max_value = max_value
        self.error_tolerance = error_tolerance

    def _check_input(self, X):
        if len(X.shape) != 2:
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))
        if self.require_symmetric_solution and (X.shape[0] != X.shape[1]):
            raise ValueError(
                "Expected square matrix, got %s array" % (X.shape,))

    def _check_missing_value_mask(self, missing):
        if not missing.any():
            raise ValueError("Input matrix is not missing any values")
        if missing.all():
            raise ValueError("Input matrix must have some non-missing values")

    def _constraints(self, X, S, error_tolerance):
        """
        Parameters
        ----------
        X : np.array
            Data matrix with missing values
        S : cvxpy.Variable
            Representation of solution variable
        """
        missing_values = np.isnan(X)
        self._check_missing_value_mask(missing_values)
        # copy the array before modifying it
        X = X.copy()
        # zero out the NaN values
        X[missing_values] = 0
        ok_mask = ~missing_values

        masked_X = cvxpy.mul_elemwise(ok_mask, X)
        masked_S = cvxpy.mul_elemwise(ok_mask, S)
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
        S = cvxpy.Variable(m, n, name="S")
        norm = cvxpy.norm(S, "nuc")
        objective = cvxpy.Minimize(norm)
        return S, objective

    def _get_solution(self, S):
        """
        Get the solution data from the cvxpy Variable and threshold to stay
        within the desired range of values.
        """
        result = S.value
        if self.min_value is not None:
            result[result < self.min_value] = self.min_value
        if self.max_value is not None:
            result[result > self.max_value] = self.max_value
        return result

    def complete(
            self,
            X,
            fast_but_approximate=True,
            verbose=True):

        """
        Expects 2d float matrix with NaN entries signifying missing values

        Returns completed matrix without any NaNs.
        """
        self._check_input(X)
        m, n = X.shape
        S, objective = self._create_objective(m, n)
        constraints = self._constraints(
            X=X,
            S=S,
            error_tolerance=self.error_tolerance)
        problem = cvxpy.Problem(objective, constraints)
        problem.solve(
            verbose=True,
            # SCS solver is known to be faster but less exact
            solver=cvxpy.SCS if fast_but_approximate else None)
        return self._get_solution(S)
