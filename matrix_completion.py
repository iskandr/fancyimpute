import cvxpy
import numpy as np


def complete(
        X,
        error_tolerance=0.0001,
        require_symmetric_solution=False,
        lower_bound=None,
        upper_bound=None,
        solver=cvxpy.SCS,
        verbose=True):
    """
    Expects 2d float matrix with NaN entries signifying missing values

    Returns completed matrix without any NaNs.
    """
    n = len(X)
    if len(X.shape) != 2:
        raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))
    if X.shape[1] != n:
        raise ValueError("Expected square matrix, got %s array" % (X.shape,))
    nan_mask = np.isnan(X)
    if not nan_mask.any():
        raise ValueError("Input matrix is not missing any values")
    if nan_mask.all():
        raise ValueError("Input matrix must have some non-missing values")

    ok_mask = (~nan_mask).astype(int)
    # copy the array before modifying it
    X = X.copy()
    X[nan_mask] = 0
    # S is the completed matrix
    S = cvxpy.Variable(n, n, name="S")
    norm = cvxpy.norm(S, "nuc")
    objective = cvxpy.Minimize(norm)

    masked_S = cvxpy.mul_elemwise(ok_mask, S)
    masked_X = cvxpy.mul_elemwise(ok_mask, X)
    sqr_diff = (masked_S - masked_X) ** 2
    close_to_data = sqr_diff <= error_tolerance

    constraints = [close_to_data]

    if require_symmetric_solution:
        constraints.append(S == S.T)

    if lower_bound is not None:
        constraints.append(S >= lower_bound)

    if upper_bound is not None:
        constraints.append(S <= upper_bound)

    problem = cvxpy.Problem(objective, constraints)
    problem.solve(verbose=True, solver=cvxpy.SCS)

    result = S.value
    if lower_bound is not None:
        result[result < lower_bound] = lower_bound
    if upper_bound is not None:
        result[result > upper_bound] = upper_bound
    return result
