from fancyimpute import NuclearNormMinimization
import numpy as np

from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error


def create_rank1_data(symmetric=False):
    """
    Returns 5x5 rank1 matrix with missing element at index (1, 2)
    """
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([0.1, -0.1, 0.2, -0.2, 0.02])
    XY = np.outer(x, y)
    XY_missing = XY.copy()
    # drop one entry
    XY_missing[1, 2] = np.nan

    if not symmetric:
        return XY, XY_missing

    # make a symmetric matrix
    XYXY = XY.T.dot(XY)

    # drop one entry
    XYXY_missing = XYXY.copy()
    XYXY_missing[1, 2] = np.nan
    return XYXY, XYXY_missing


def test_rank1_convex_solver():
    XY_rank1, XY_missing_rank1 = create_rank1_data(symmetric=False)
    XY_completed_rank1 = NuclearNormMinimization().complete(XY_missing_rank1)
    assert abs(XY_completed_rank1[1, 2] - XY_rank1[1, 2]) < 0.001, \
        "Expected %0.4f but got %0.4f" % (
            XY_rank1[1, 2], XY_completed_rank1[1, 2])


def test_rank1_symmetric_convex_solver():
    XYXY_rank1, XYXY_missing_rank1 = create_rank1_data(symmetric=True)
    solver = NuclearNormMinimization(require_symmetric_solution=True)
    completed = solver.complete(XYXY_missing_rank1)
    assert abs(completed[1, 2] - XYXY_rank1[1, 2]) < 0.001, \
        "Expected %0.4f but got %0.4f" % (
            XYXY_rank1[1, 2], completed[1, 2])


def test_nuclear_norm_minimization_with_low_rank_random_matrix():
    solver = NuclearNormMinimization(require_symmetric_solution=False)
    XY_completed = solver.complete(XY_incomplete[:100])
    _, missing_mae = reconstruction_error(
        XY[:100], XY_completed, missing_mask[:100], name="NuclearNorm")
    assert missing_mae < 0.1, "Error too high!"

if __name__ == "__main__":
    test_rank1_convex_solver()
    test_rank1_symmetric_convex_solver()
    test_nuclear_norm_minimization_with_low_rank_random_matrix()
