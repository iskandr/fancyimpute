from fancyimpute import Solver, SimpleFill

from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error

import numpy as np
import warnings


def test_prepare_input_data():
    _solver = Solver()
    print(_solver) # for improved coverage
    # test that a complete matrix returns a warning
    X1 = np.zeros((5, 5))
    with warnings.catch_warnings(record=True) as w:
        _solver.prepare_input_data(X1)
        assert str(w[0].message) == "Input matrix is not missing any values", "Warning is not generated for a complete matrix"
    # test that an incomplete matrix does not return a warning
    X2 = np.zeros((5, 5))
    X2[2, 3] = None
    with warnings.catch_warnings(record=True) as w:
        _solver.prepare_input_data(X2)
        assert len(w) == 0, "Warning is generated for an incomplete matrix"


def test_solver_fill_methods_with_low_rank_random_matrix():
    for fill_method in ("zero", "mean", "median", "min", "random"):
        imputer = SimpleFill(fill_method=fill_method)
        XY_completed = imputer.fit_transform(XY_incomplete)
        _, missing_mae = reconstruction_error(
            XY,
            XY_completed,
            missing_mask,
            name="Solver with fill_method=%s" %fill_method)
        assert missing_mae < 5, "Error too high for Solver with %s fill method!" %fill_method


if __name__ == "__main__":
    test_prepare_input_data()
    test_solver_fill_methods_with_low_rank_random_matrix()