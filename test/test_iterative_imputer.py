from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error

import numpy as np
from fancyimpute import IterativeImputer


def test_iterative_imputer_with_low_rank_random_matrix():
    imputer = IterativeImputer(n_iter=50, random_state=0)
    XY_completed = imputer.complete(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="IterativeImputer")
    assert missing_mae < 0.1, "Error too high with IterativeImputer method!"


def test_iterative_imputer_with_low_rank_random_matrix_approximate():
    imputer = IterativeImputer(n_iter=50, n_nearest_features=5, random_state=0)
    XY_completed = imputer.complete(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="IterativeImputer with n_nearest_features=5")
    assert missing_mae < 0.1, "Error too high with IterativeImputer " \
                              "method using n_nearest_features=5!"


def test_iterative_imputer_as_mice_with_low_rank_random_matrix_approximate():
    n_imputations = 5
    XY_completed = []
    for i in range(n_imputations):
        imputer = IterativeImputer(n_iter=5, sample_posterior=True, random_state=i)
        XY_completed.append(imputer.complete(XY_incomplete))
    _, missing_mae = reconstruction_error(
        XY,
        np.mean(XY_completed, axis=0),
        missing_mask,
        name="IterativeImputer as MICE")
    assert missing_mae < 0.1, "Error too high with IterativeImputer as MICE!"


if __name__ == "__main__":
    test_iterative_imputer_with_low_rank_random_matrix()
    test_iterative_imputer_with_low_rank_random_matrix_approximate()
    test_iterative_imputer_as_mice_with_low_rank_random_matrix_approximate()