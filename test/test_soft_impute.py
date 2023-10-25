from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error
import numpy as np
from fancyimpute import SoftImpute


def test_soft_impute_with_low_rank_random_matrix():
    solver = SoftImpute()
    XY_completed = solver.fit_transform(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="SoftImpute")
    assert missing_mae < 0.1, "Error too high!"


# test if the solver for a submodel is running for a numpy array without any missing data
def check_for_no_missing_data():
    X = np.ones((5, 5))
    Xf = SoftImpute().fit_transform(X)
    assert (Xf.all() == X.all())


if __name__ == "__main__":
    test_soft_impute_with_low_rank_random_matrix()
    check_for_no_missing_data()
