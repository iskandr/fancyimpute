import warnings

import numpy as np
from fancySVD import Solver


def test_prepare_input_data():
    _solver = Solver()
    print(_solver)  # for improved coverage
    # test that a complete matrix returns a warning
    X1 = np.zeros((5, 5))
    with warnings.catch_warnings(record=True) as w:
        _solver.prepare_input_data(X1)
        assert (
            str(w[0].message) == "Input matrix is not missing any values"
        ), "Warning is not generated for a complete matrix"
    # test that an incomplete matrix does not return a warning
    X2 = np.zeros((5, 5))
    X2[2, 3] = None
    with warnings.catch_warnings(record=True) as w:
        _solver.prepare_input_data(X2)
        assert len(w) == 0, "Warning is generated for an incomplete matrix"


if __name__ == "__main__":
    test_prepare_input_data()
