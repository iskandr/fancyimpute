from fancyimpute import MatrixFactorization

from low_rank_data import XY, XY_incomplete, missing_mask
from common import reconstruction_error


def test_matrix_factorization_with_low_rank_random_matrix():
    solver = MatrixFactorization(
        learning_rate=0.01,
        rank=3,
        l2_penalty=0,
        min_improvement=1e-6)
    XY_completed = solver.fit_transform(XY_incomplete)
    _, missing_mae = reconstruction_error(
        XY,
        XY_completed,
        missing_mask,
        name="MatrixFactorization")
    assert missing_mae < 0.1, "Error too high!"

if __name__ == "__main__":
    test_matrix_factorization_with_low_rank_random_matrix()
