from fancyimpute import MatrixFactorization

from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error
from random_initialization import initialize_random_seed


def test_matrix_factorization_with_low_rank_random_matrix():
    initialize_random_seed()  # for reproducibility
    solver = MatrixFactorization(
        learning_rate=0.01,
        rank=3,
        l2_penalty=0,
        min_improvement=1e-6,
        verbose=False)
    XY_completed = solver.fit_transform(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MatrixFactorization")
    assert missing_mae < 0.1, "Error too high!"

    initialize_random_seed()  # for reproducibility
    solver = MatrixFactorization(
        learning_rate=0.01,
        rank=3,
        l2_penalty=0,
        min_improvement=1e-6,
        verbose=False)
    XY_completed = solver.fit(XY_incomplete, missing_mask)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MatrixFactorization")
    assert missing_mae < 0.1, "Error too high!"

    XY_completed = solver.transform(XY_incomplete, missing_mask)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MatrixFactorization")
    assert missing_mae < 0.1, "Error too high!"

if __name__ == "__main__":
    test_matrix_factorization_with_low_rank_random_matrix()
