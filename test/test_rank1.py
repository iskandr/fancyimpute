from fancyimpute import ConvexSolver
import numpy as np


def test_rank1_outer_product():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([0.1, -0.1, 0.2, -0.2, 0.02])
    XY = np.outer(x, y)
    XY_missing = XY.copy()

    # drop one entry
    XY_missing[1, 2] = np.nan

    XY_completed = ConvexSolver().complete(XY_missing)
    assert abs(XY_completed[1, 2] - XY[1, 2]) < 0.001, \
        "Expected %0.4f but got %0.4f" % (
            XY[1, 2], XY_completed[1, 2])


def test_rank1_symmetric():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([0.1, -0.1, 0.2, -0.2, 0.02])
    XY = np.outer(x, y)
    # make a symmetric matrix
    XYXY = XY.T.dot(XY)

    # drop one entry
    missing = XYXY.copy()
    missing[1, 2] = np.nan

    completed = ConvexSolver(require_symmetric_solution=True).complete(missing)
    assert abs(completed[1, 2] - XYXY[1, 2]) < 0.001, \
        "Expected %0.4f but got %0.4f" % (
            XYXY[1, 2], completed[1, 2])
